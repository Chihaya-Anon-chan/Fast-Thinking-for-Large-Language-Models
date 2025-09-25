# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import weakref

import peft
from peft.utils import _get_submodules

# Import official 4D mask conversion method
try:
    from models.base_llms.qwen3.modeling_qwen3 import Qwen3Model
except ImportError:
    Qwen3Model = None

from .config import CodebookConfig
from .layer import CodebookAttention, ThinkingRefiner
from .utils import is_codebook_trainable, disable_early_lora

import os



def _ensure_length(mask_2d, K, val=1.0):
    """Ensure 2D mask length matches target K dimension"""
    if mask_2d.shape[1] < K:
        mask_2d = F.pad(mask_2d, (0, K - mask_2d.shape[1]), value=val)
    elif mask_2d.shape[1] > K:
        mask_2d = mask_2d[:, :K]
    return mask_2d


class _FullVisibilityAttnWrapper(nn.Module):
    """
    Full visibility attention wrapper: automatically converts 2D full_attention_mask to 4D causal mask
    Used for inserted layer and all layers after it, ensuring thinking tokens are always visible
    """
    def __init__(self, inner_attn: nn.Module, adapter: "CodebookAdapterModel"):
        super().__init__()
        self.inner = inner_attn
        self.adapter_ref = weakref.ref(adapter)  # Use weak reference to avoid circular reference

    def forward(self, hidden_states, attention_mask=None, cache_position=None,
                past_key_value=None, **kwargs):
        # Get full_attention_mask from runtime context
        adapter = self.adapter_ref()
        if adapter is None:
            return self.inner(hidden_states=hidden_states, attention_mask=attention_mask,
                            cache_position=cache_position, past_key_value=past_key_value, **kwargs)
        
        fam_2d = adapter._runtime_ctx.get("full_mask_2d", None)
        
        
        if fam_2d is not None and attention_mask is not None and attention_mask.dim() == 4 and Qwen3Model is not None:
            B, _, Q, K = attention_mask.shape
            # Ensure 2D mask length matches K dimension
            fam_2d = _ensure_length(fam_2d.to(hidden_states.device, hidden_states.dtype), K, 1.0)
            
            # Use official method to convert to 4D causal mask
            attention_mask = Qwen3Model._prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask=fam_2d, 
                sequence_length=Q, 
                target_length=K,
                dtype=hidden_states.dtype, 
                cache_position=cache_position,
                batch_size=B, 
                config=adapter.model.config, 
                past_key_values=past_key_value,
            )
        
        # Call original attention
        attn_output, attn_weights = self.inner(hidden_states=hidden_states, attention_mask=attention_mask,
                                              cache_position=cache_position, past_key_value=past_key_value, **kwargs)
        
        # Apply ThinkingRefiner (if exists)
        _ref = getattr(self, "_refiner_ref", None)
        refiner = _ref() if _ref is not None else None
        if refiner is not None:
            input_mask_2d = adapter._runtime_ctx.get("input_mask_2d", None)
            if input_mask_2d is not None:
                
                # Actually call refiner
                attn_output = refiner(attn_output, input_mask_2d)
                
        
        return attn_output, attn_weights

class CodebookAdapterModel(nn.Module):

    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.peft_config: CodebookConfig = config
        self._cached_adapters: Dict[str, List] = {}
        self._runtime_ctx = {"full_mask_2d": None, "input_mask_2d": None, "last_K": None, "prefill_seen": False}  # Runtime context
        self.add_adapter(config)
        self._mark_only_adaption_prompts_as_trainable(self.model)


    def save_pretrained(self, save_directory):

        peft_config = self.peft_config

        # save only the trainable weights ***excluding*** LoRA params (they will be saved by PEFT separately)
        state_dict = self.state_dict()
        output_state_dict = {
            k: state_dict[k]
            for k in state_dict.keys()
            if is_codebook_trainable(k) and "lora_" not in k  # filter out LoRA weights
        }

        os.makedirs(save_directory, exist_ok=True)

        # 1) Save Codebook / Refiner weights
        pth_file = os.path.join(save_directory, "adapter_model.pth")
        torch.save(output_state_dict, pth_file)

        # 2) Save Codebook config (inherits from PeftConfig)
        inference_mode = peft_config.inference_mode
        peft_config.inference_mode = True

        if peft_config.task_type is None:
            # deal with auto mapping
            base_model_class = self.model.__class__
            parent_library = base_model_class.__module__

            auto_mapping_dict = {
                "base_model_class": base_model_class.__name__,
                "parent_library": parent_library,
            }

        peft_config.save_pretrained(save_directory, auto_mapping_dict=auto_mapping_dict)

        peft_config.inference_mode = inference_mode

        # 3) Save LoRA adapter using PEFT – this will write its own adapter_config.json & weights
        lora_dir = os.path.join(save_directory, "lora")
        self.model.save_pretrained(lora_dir)

    @classmethod
    def from_pretrained(cls, model_id, model_path, is_trainable=False):
        config = CodebookConfig.from_pretrained(model_path)

        # -------------------------------------------------------------
        # Step 1: (Optional) Restore LoRA adapter if it exists
        # -------------------------------------------------------------
        lora_path = os.path.join(model_path, "lora")
        if os.path.exists(lora_path):
            # Load LoRA weights onto the base model first
            base_model = peft.PeftModel.from_pretrained(model_id, lora_path, is_trainable=is_trainable)
        else:
            # Fallback to the raw base model (no LoRA weights found)
            base_model = model_id

        # -------------------------------------------------------------
        # Step 2: Wrap with CodebookAdapterModel (adds Codebook & Refiner)
        # -------------------------------------------------------------
        model = cls(base_model, config)

        # -------------------------------------------------------------
        # Step 3: Load Codebook / Refiner weights
        # -------------------------------------------------------------
        # Get model device
        model_device = next(model.model.parameters()).device
        pth_file = os.path.join(model_path, "adapter_model.pth")

        if os.path.exists(pth_file):
            adapters_weights = torch.load(pth_file, map_location="cpu")

            # Transfer weights to the device where the model is located
            adapters_weights_on_device = {}
            for key, value in adapters_weights.items():
                if isinstance(value, torch.Tensor):
                    adapters_weights_on_device[key] = value.to(device=model_device)
                else:
                    adapters_weights_on_device[key] = value

            incompat = model.load_state_dict(adapters_weights_on_device, strict=False)
            
            # LoRA weight check (only check inserted layer and layers after)
            def check_lora_layers():
                lora_layers = []
                for n, p in model.named_parameters():
                    if "lora_" in n and ".layers." in n:
                        try:
                            layer_idx = int(n.split(".layers.")[1].split(".")[0])
                            if layer_idx >= config.inserted_layer - 1:  # 0-indexed
                                lora_layers.append((layer_idx, n, float(p.data.norm().item())))
                        except:
                            continue
                return lora_layers
            

        # -------------------------------------------------------------
        # Step 4: Freeze params if not trainable
        # -------------------------------------------------------------
        if not is_trainable:
            for _, p in model.named_parameters():
                if p.requires_grad:
                    p.requires_grad = False

        # Disable early-layer LoRA then merge remaining ones into backbone
        disable_early_lora(model, config.inserted_layer)

        # If underlying is PeftModel, merge & unload to eliminate runtime LoRA layers
        try:
            base_peft = model.model  # CodebookAdapterModel.model -> PeftModel
            if hasattr(base_peft, "merge_and_unload"):
                base_peft.merge_and_unload()
        except Exception as e:
            pass

        return model

    def generate(
            self, 
            input_ids, 
            attention_mask, 
            eos_token_id, 
            max_new_tokens=1024, 
            **kwargs
        ):
        generated = input_ids
        attention_mask = attention_mask
        full_attention_mask = kwargs['full_attention_mask']
        input_mask = kwargs['input_mask']
        past_key_values = None

        # reset runtime counters for this round
        self._runtime_ctx["last_K"] = None
        self._runtime_ctx["fam_warned"] = False
        self._runtime_ctx["inject_cnt"] = 0
        self._runtime_ctx["prefill_seen"] = False
        self._runtime_ctx["decode_chk_printed"] = False

        for _ in range(max_new_tokens): 
            torch.cuda.empty_cache()

            if past_key_values is not None:
                # During incremental generation, only use the latest token
                input_ids = generated[:, -1].unsqueeze(-1)

            outputs = self(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                input_mask=input_mask,
                full_attention_mask=full_attention_mask,
                past_key_values=past_key_values,
            )
            next_token_logits = outputs.logits[:, -1, :]

            past_key_values = outputs['past_key_values']

            # Fix: Expand by batch dimension, support batch>1
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)  # [B, 1]

            generated = torch.cat([generated, next_token], dim=-1)
            
            # Correctly expand three types of masks by batch dimension
            B = generated.size(0)
            device = attention_mask.device
            
            one_f = torch.ones(B, 1, dtype=attention_mask.dtype, device=device)
            one_full = torch.ones(B, 1, dtype=full_attention_mask.dtype, device=device)
            zero_m = torch.zeros(B, 1, dtype=input_mask.dtype, device=device)
            
            attention_mask = torch.cat([attention_mask, one_f], dim=-1)
            full_attention_mask = torch.cat([full_attention_mask, one_full], dim=-1)
            input_mask = torch.cat([input_mask, zero_m], dim=-1)

            # Fix EOS termination condition: support int or list/tuple eos_token_id
            if isinstance(eos_token_id, int):
                hit_eos = (next_token == eos_token_id).any()
            else:
                hit_eos = torch.isin(next_token, torch.tensor(eos_token_id, device=next_token.device)).any()
            if hit_eos:
                break


        return generated

    
    def add_adapter(self, config: CodebookConfig) -> None:

        parents = []
        for name, _ in self.model.named_modules():
            if name.endswith('self_attn'):
                par, _, _ = _get_submodules(self.model, name)
                parents.append(par)
        if len(parents) < config.inserted_layer:
            raise ValueError(
                f"Config specifies more adapter layers '{config.inserted_layer}'"
                f" than the model has '{len(parents)}'."
            )

        codebook_parent = parents[config.inserted_layer - 1]
        self._create_adapted_attentions(config, codebook_parent)

        for layer_idx in range(config.inserted_layer - 1, len(parents)):
            parent = parents[layer_idx]
            inner = getattr(parent, 'self_attn')
            wrapped = _FullVisibilityAttnWrapper(inner, adapter=self)
            wrapped._refiner_ref = None
            setattr(parent, 'self_attn', wrapped)

        self._add_thinking_refiners(config, parents)

        for layer_idx in range(config.inserted_layer, len(parents)):
            parent = parents[layer_idx]
            if hasattr(parent, 'thinking_refiner'):
                wrapped_attn = getattr(parent, 'self_attn')
                if isinstance(wrapped_attn, _FullVisibilityAttnWrapper):
                    wrapped_attn._refiner_ref = weakref.ref(parent.thinking_refiner)

    def _create_adapted_attentions(self, config: CodebookConfig, parents: nn.Module) -> None:
        """Wrap Attention modules with newly created CodebookAttention modules."""
        attn = CodebookAttention(
            select_len=config.select_len,
            codebook_size=config.codebook_size,
            model=getattr(parents, 'self_attn'),
        )
        # Set adapter reference for accessing runtime_ctx
        attn._adapter_ref = weakref.ref(self)
        setattr(parents, 'self_attn', attn)
        
    def _add_thinking_refiners(self, config: CodebookConfig, all_parents: List) -> None:
        """Add ThinkingRefiner to all layers after insertion layer"""
        # Get hidden_size from model config
        hidden_size = self.model.config.hidden_size

        # Get model data type, inferred from first parameter
        model_dtype = next(self.model.parameters()).dtype

        # Calculate number of layers needing refiner
        num_layers_after_insertion = len(all_parents) - config.inserted_layer

        # Get model device
        model_device = next(self.model.parameters()).device

        for i, layer_idx in enumerate(range(config.inserted_layer, len(all_parents))):
            layer_parent = all_parents[layer_idx]

            # Create ThinkingRefiner, using data type consistent with model
            refiner = ThinkingRefiner(
                hidden_size=hidden_size,
                dtype=model_dtype
            )

            # Ensure ThinkingRefiner is on correct device
            refiner = refiner.to(device=model_device, dtype=model_dtype)

            # Add refiner to layer_parent
            setattr(layer_parent, 'thinking_refiner', refiner)

    def _mark_only_adaption_prompts_as_trainable(self, model: nn.Module) -> None:
        """Freeze all parameters of the model except the Codebook."""
        for n, p in model.named_parameters():
            if not is_codebook_trainable(n):
                p.requires_grad = False

    def forward(self, *args, **kwargs):
        """Delegate forward pass to the wrapped base model while keeping the full
        CausalLMOutput structure (logits + hidden_states + past_key_values …)."""
        # Write mask information to runtime context
        fam = kwargs.get("full_attention_mask", None)
        input_mask = kwargs.get("input_mask", None)
        if fam is not None:
            self._runtime_ctx["full_mask_2d"] = fam
        if input_mask is not None:
            self._runtime_ctx["input_mask_2d"] = input_mask
        try:
            return self.model(*args, **kwargs)
        finally:
            self._runtime_ctx["full_mask_2d"] = None
            self._runtime_ctx["input_mask_2d"] = None

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            # This is necessary as e.g. causal models have various methods that we
            # don't want to re-implement here.
            if name == "model":  
                raise
            return getattr(self.model, name)
