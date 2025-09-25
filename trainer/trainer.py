import transformers
from transformers import PreTrainedModel
from transformers import Trainer, TrainingArguments, TrainerCallback
import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from typing import Union, Optional, Dict, List
from transformers.trainer_pt_utils import LabelSmoother
import os
import json


class CustomTrainer(Trainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", nn.Module] = None,
        ref_model: Union["PreTrainedModel", nn.Module] = None,
        args: "TrainingArguments" = None,
        data_collator: Optional["DataCollator"] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        mode: Optional[str] = "alignment",
        callbacks: Optional[List[TrainerCallback]] = None,
        use_lr_multipliers: bool = False,
        lr_config_path: str = "lr_multipliers.json",
        inserted_layer: Optional[int] = None,
    ):
        # Save lr multiplier config
        self.use_lr_multipliers = use_lr_multipliers
        self.lr_config_path = lr_config_path
        self.inserted_layer = inserted_layer
        
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            callbacks=callbacks
        )

        self.ref_model = ref_model
        self.label_smoother = LabelSmoother(epsilon=0.1)
        self.mode = mode
        
        # Dynamic weight adjustment parameters
        self.total_training_steps = None  # Will be set at training start
        self.current_step = 0

    def _get_dynamic_loss_weights(self):
        """
        Calculate dynamic loss weights: gradually transition from fully relying on inserted layer to including final layer
        
        Returns:
            alpha: inserted layer weight (gradually decrease from 1.0 to 0.5)
            beta: final layer weight (gradually increase from 0.0 to 0.5)
        """
        if self.total_training_steps is None:
            # Use HuggingFace Trainer's standard method to calculate total steps
            # Important: Consider world_size in DDP mode
            try:
                # Get world_size (DDP environment variable)
                world_size = int(os.environ.get("WORLD_SIZE", 1))
                
                # Method 1: Use get_train_dataloader for accurate dataloader
                train_dataloader = self.get_train_dataloader()
                num_batches_per_epoch = len(train_dataloader)
                self.total_training_steps = (
                    num_batches_per_epoch * self.args.num_train_epochs
                )
                
                # Note: HuggingFace Trainer already handles DDP sharding in get_train_dataloader
                # So num_batches_per_epoch here is the number of batches each GPU sees
                
            except:
                # Method 2: Fallback calculation method
                world_size = int(os.environ.get("WORLD_SIZE", 1))
                dataset_length = len(self.train_dataset) if self.train_dataset else 1
                
                # In DDP mode, each GPU only processes 1/world_size of the data
                effective_dataset_length = dataset_length // world_size
                
                effective_batch_size = (
                    self.args.per_device_train_batch_size * 
                    self.args.gradient_accumulation_steps
                )
                steps_per_epoch = effective_dataset_length // effective_batch_size
                self.total_training_steps = steps_per_epoch * self.args.num_train_epochs
            
        # Calculate training progress (0.0 to 1.0)
        progress = min(self.current_step / self.total_training_steps, 1.0)
        
        # Dynamic weight adjustment strategy - new version: 30%-40%-30%
        if progress <= 0.3:
            # First 30%: completely rely on inserted layer (basic learning period)
            alpha = 1.0
            beta = 0.0
        elif progress <= 0.7:
            # 30%-70%: linear transition period (40% time)
            transition_progress = (progress - 0.3) / 0.4  # normalize to [0,1]
            alpha = 1.0 - 0.5 * transition_progress  # transition from 1.0 to 0.5
            beta = 0.5 * transition_progress          # transition from 0.0 to 0.5
        else:
            # Last 30%: stable at balanced weights
            alpha = 0.5
            beta = 0.5
            
        return alpha, beta

    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling function, considering attention mask"""
        token_embeddings = model_output
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=torch.finfo(input_mask_expanded.dtype).smallest_normal
        )



    def _compute_layer_alignment_loss(self, main_hidden, ref_hidden, main_input_mask, ref_thinking_mask):
        """Calculate single-layer alignment loss - using cosine similarity loss"""

        # Extract thinking vectors generated by codebook
        thinking_positions = (main_input_mask == 2).float()
        main_thinking_vectors = main_hidden * thinking_positions.unsqueeze(-1)
        main_thinking_pooled = self._mean_pooling(main_thinking_vectors, thinking_positions)

        # Extract vectors from real thinking text
        ref_thinking_positions = (ref_thinking_mask == 1).float()
        ref_thinking_vectors = ref_hidden * ref_thinking_positions.unsqueeze(-1)
        ref_thinking_pooled = self._mean_pooling(ref_thinking_vectors, ref_thinking_positions)

        # Training info logging (once per training)
        if not hasattr(self, '_loss_debug_logged'):
            print(f"Cosine loss setup:")
            print(f"   Main thinking positions: {thinking_positions.sum().item()}")
            print(f"   Reference thinking positions: {ref_thinking_positions.sum().item()}")
            print(f"   Vector shape: {main_thinking_pooled.shape}")
            self._loss_debug_logged = True

        # Calculate cosine similarity loss
        cosine_sim = F.cosine_similarity(main_thinking_pooled, ref_thinking_pooled, dim=-1)
        
        return 1 - cosine_sim.mean()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss, handling DDP wrapper."""
        # Unwrap DDP if necessary
        base_model = model.module if hasattr(model, "module") else model
        model = base_model  # use base_model for rest of logic
        
        # Skip detailed parameter checks for cleaner output
        
        # Main sequence (codebook version)
        main_input_ids = inputs["input_ids"][0::2, ...]
        main_attention_mask = inputs["attention_mask"][0::2, ...]
        main_full_attention_mask = inputs["full_attention_mask"][0::2, ...]
        main_labels = inputs["labels"][0::2, ...]
        main_input_mask = inputs["mask"][0::2, ...]
        
        # Reference sequence (real thinking text)
        ref_input_ids = inputs["input_ids"][1::2, ...]
        ref_attention_mask = inputs["attention_mask"][1::2, ...]
        ref_thinking_mask = inputs["mask"][1::2, ...]

        # Main model forward pass (using codebook)
        main_outputs = model(
            input_ids=main_input_ids,
            labels=main_labels,
            attention_mask=main_attention_mask,
            output_hidden_states=True,
            return_dict=True,
            input_mask=main_input_mask,
            full_attention_mask=main_full_attention_mask
        )

        loss = 0

        if self.mode == 'alignment':
            # Reference model forward pass (real thinking text)
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=ref_input_ids,
                    attention_mask=ref_attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )

            # Get inserted layer index
            # Note: we want to supervise the output of the inserted layer, not the input
            # inserted_layer=N means layer N injection, we need layer N output (index N)
            inserted_layer_idx = base_model.peft_config.inserted_layer
            
            # Ensure valid layer index
            if inserted_layer_idx >= len(main_outputs.hidden_states):
                inserted_layer_idx = len(main_outputs.hidden_states) - 1
            
            # 1. Inserted layer loss: supervise initial generation of thinking vectors
            inserted_layer_loss = self._compute_layer_alignment_loss(
                main_outputs.hidden_states[inserted_layer_idx],
                ref_outputs.hidden_states[inserted_layer_idx],
                main_input_mask,
                ref_thinking_mask
            )
            
            # 2. Subsequent layer loss: supervise semantic alignment of all layers after insertion
            later_layer_losses = []
            for layer_idx in range(inserted_layer_idx + 1, len(main_outputs.hidden_states)):
                layer_loss = self._compute_layer_alignment_loss(
                    main_outputs.hidden_states[layer_idx],
                    ref_outputs.hidden_states[layer_idx],
                    main_input_mask,
                    ref_thinking_mask
                )
                later_layer_losses.append(layer_loss)

            # If subsequent layers exist, take average; otherwise degrade to using only last layer (compatible with old logic)
            if len(later_layer_losses) > 0:
                final_layer_loss = torch.stack(later_layer_losses).mean()
            else:
                final_layer_loss = self._compute_layer_alignment_loss(
                    main_outputs.hidden_states[-1],
                    ref_outputs.hidden_states[-1],
                    main_input_mask,
                    ref_thinking_mask
                )
            
            # Dynamic weighted loss combination: gradually transition from complete reliance on inserted layer to balance
            alpha, beta = self._get_dynamic_loss_weights()
            loss = alpha * inserted_layer_loss + beta * final_layer_loss

            # Log training progress (every 100 steps)
            if self.current_step % 100 == 0:
                progress = min(self.current_step / self.total_training_steps, 1.0)
                print(f"Step {self.current_step} ({progress*100:.1f}%): loss={loss:.4f} (α={alpha:.2f}, β={beta:.2f})")

                # Check for potential training issues
                if inserted_layer_loss >= 0.99:
                    print(f"Warning: inserted layer loss too high ({inserted_layer_loss:.3f}), check learning rate/codebook initialization")

            # Update step counter
            self.current_step += 1

        elif self.mode == 'sft':
            # Standard language modeling loss (no additional weighting)
            loss = main_outputs.loss

        if return_outputs:
            outputs = {"hidden_states": main_outputs.hidden_states[-1]}
            return (loss, outputs)
        
        return loss

    def _check_trainable_parameters(self):
        """Check trainable parameter status"""
        from models.codebook.utils import is_codebook_trainable

        print(f"\nTrainable parameter check:")
        trainable_params = 0
        total_params = 0
        codebook_params = 0
        lora_norms = []

        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                if is_codebook_trainable(name):
                    codebook_params += param.numel()
                    # Codebook parameter is trainable
            else:
                if is_codebook_trainable(name):
                    print(f"Warning: {name} not set as trainable!")

            # Collect LoRA param norms for quick diagnostics
            if "lora_A" in name or "lora_B" in name:
                try:
                    lora_norms.append(param.data.norm().item())
                except Exception:
                    pass
        
        print(f"Total: {total_params:,}, Trainable: {trainable_params:,}, Codebook: {codebook_params:,}")
        
        if lora_norms:
            mean_norm = sum(lora_norms) / len(lora_norms)
            max_norm = max(lora_norms)
            print(f"LoRA ΔW  mean‖·‖={mean_norm:.4f}, max‖·‖={max_norm:.4f}")
        
        if codebook_params == 0:
            print(f"Error: No trainable codebook parameters found!")
            print(f"This explains why grad_norm=0.0 and inserted_layer_loss=1.0")
            print(f"Attempting to reset trainable parameters...")
            self._force_reset_trainable_parameters()
        
        print(f"========================\n")

    def _force_reset_trainable_parameters(self):
        """Force reset trainable parameters"""
        from models.codebook.utils import is_codebook_trainable

        print(f"Force resetting trainable parameters...")
        reset_count = 0

        # First freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Then only unfreeze codebook-related parameters
        for name, param in self.model.named_parameters():
            if is_codebook_trainable(name):
                param.requires_grad = True
                reset_count += 1
                print(f"  Set as trainable: {name}")

        print(f"Reset {reset_count} Codebook parameters as trainable")

        # Check parameters again
        print(f"Rechecking parameter status...")
        self._check_trainable_parameters()

    def _check_model_structure(self):
        """Check model structure and confirm Codebook installation"""
        print(f"\n=== Model Structure Check ===")

        # Check inserted layer configuration
        inserted_layer = self.model.peft_config.inserted_layer
        print(f"Configured inserted layer: {inserted_layer}")

        # Check model layer count
        if hasattr(self.model.model, 'layers'):
            total_layers = len(self.model.model.layers)
            print(f"Total model layers: {total_layers}")

            # Check if inserted layer has CodebookAttention
            target_layer_idx = inserted_layer - 1  # Convert to 0-indexed
            if target_layer_idx < total_layers:
                target_layer = self.model.model.layers[target_layer_idx]
                has_codebook = hasattr(target_layer.self_attn, 'model') and hasattr(target_layer.self_attn, 'codebook')
                print(f"Layer {inserted_layer} has Codebook: {has_codebook}")

                if has_codebook:
                    codebook_size = target_layer.self_attn.codebook_size
                    select_len = target_layer.self_attn.select_len
                    print(f"   Codebook size: {codebook_size}")
                    print(f"   Selection length: {select_len}")
                else:
                    print(f"   Warning: Layer {inserted_layer} has no Codebook!")
                    print(f"   Layer type: {type(target_layer.self_attn)}")
            else:
                print(f"   Warning: Inserted layer index out of range!")

        # Check ThinkingRefiner
        refiner_count = 0
        for i in range(inserted_layer, total_layers):
            layer = self.model.model.layers[i]
            if hasattr(layer, 'thinking_refiner'):
                refiner_count += 1

        print(f"Installed ThinkingRefiner count: {refiner_count}")
        print(f"Expected ThinkingRefiner count: {total_layers - inserted_layer}")
        print(f"========================\n")

    def on_train_begin(self):
        """Training initialization"""
        super().on_train_begin()
        # Reset step counter
        self.current_step = 0
        # Ensure total steps are calculated correctly
        if self.total_training_steps is None:
            self._get_dynamic_loss_weights()  # This will trigger total step calculation

        # Get DDP information
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))

        print(f"Training started: {self.total_training_steps} steps, lr={self.args.learning_rate}")

        # Quick parameter check
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable:,}")

    def create_optimizer(self):
        """
        Override optimizer creation to support component-level learning rate control
        """
        if self.use_lr_multipliers and os.path.exists(self.lr_config_path):
            print(f"\nUsing component-level learning rate configuration: {self.lr_config_path}")

            # Import lr scheduler utilities
            try:
                from trainer.lr_scheduler_utils import ComponentLRScheduler

                scheduler = ComponentLRScheduler(self.lr_config_path)

                # Determine current stage
                stage = "stage1" if self.mode == "alignment" else "stage2"

                # Get parameter groups
                param_groups = scheduler.get_parameter_groups(
                    self.model,
                    self.args.learning_rate,
                    stage=stage,
                    inserted_layer=self.inserted_layer
                )

                # Add weight_decay to each group
                for group in param_groups:
                    if 'weight_decay' not in group:
                        group['weight_decay'] = self.args.weight_decay

                # Create optimizer
                optimizer_cls = torch.optim.AdamW
                if hasattr(self.args, 'optim'):
                    if 'sgd' in self.args.optim.lower():
                        optimizer_cls = torch.optim.SGD
                    elif 'adam' in self.args.optim.lower() and 'adamw' not in self.args.optim.lower():
                        optimizer_cls = torch.optim.Adam

                self.optimizer = optimizer_cls(param_groups)

                print(f"Created optimizer with {len(param_groups)} parameter groups")
                return self.optimizer

            except ImportError as e:
                print(f"Warning: Cannot import lr_scheduler_utils: {e}")
                print("  Using default optimizer")
            except Exception as e:
                print(f"Warning: Failed to create component-level optimizer: {e}")
                print("  Using default optimizer")

        # Fallback to default optimizer creation
        return super().create_optimizer()

    # ------------------------------------------------------------------
    #  Print LoRA norms at training end to observe changes
    # ------------------------------------------------------------------

    def on_train_end(self, **kwargs):  # type: ignore[override]
        if int(os.environ.get("RANK", 0)) != 0:
            return

        print("\n=== Training Complete: LoRA Parameter Statistics ===")
        lora_norms = []
        for name, param in self.model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                try:
                    lora_norms.append(param.data.norm().item())
                except Exception:
                    pass

        if lora_norms:
            mean_norm = sum(lora_norms) / len(lora_norms)
            max_norm = max(lora_norms)
            print(f"LoRA parameters - mean norm: {mean_norm:.4f}, max norm: {max_norm:.4f}")
        else:
            print("(No LoRA parameters detected)")

