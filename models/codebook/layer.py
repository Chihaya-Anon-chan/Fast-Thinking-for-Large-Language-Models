import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import hyperfanin_init_weight, RMSNorm

try:
    from models.base_llms.qwen3.modeling_qwen3 import Qwen3Model
except ImportError as e:
    Qwen3Model = None


class ThinkingRefiner(nn.Module):
    """Lightweight thinking refinement module for processing thinking vectors"""

    def __init__(self, hidden_size: int, dtype=torch.bfloat16):
        super().__init__()
        self.hidden_size = hidden_size
        self.dtype = dtype

        self.refiner = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        for module in self.refiner:
            if isinstance(module, nn.Linear):
                fan_in, fan_out = module.weight.shape[1], module.weight.shape[0]
                gain = 0.02 if (fan_in > 1000 or fan_out > 1000) else 0.1

                if module.weight.dtype in [torch.bfloat16, torch.float16]:
                    temp_weight = torch.empty(module.weight.shape, dtype=torch.float32)
                    nn.init.orthogonal_(temp_weight, gain=gain)
                    module.weight.data = temp_weight.to(dtype=module.weight.dtype, device=module.weight.device)
                else:
                    nn.init.orthogonal_(module.weight, gain=gain)

                if module.bias is not None:
                    nn.init.uniform_(module.bias, -0.001, 0.001)

        self.alpha = nn.Parameter(torch.tensor(0.1, dtype=dtype))
        self.norm = RMSNorm(hidden_size)
        
    def forward(self, hidden_states: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        if hidden_states.size(1) != input_mask.size(1):
            return hidden_states

        thinking_mask = (input_mask == 2)

        if thinking_mask.any():
            batch_size, seq_len = thinking_mask.shape

            for batch_idx in range(batch_size):
                batch_thinking_mask = thinking_mask[batch_idx]
                if batch_thinking_mask.any():
                    thinking_indices = torch.where(batch_thinking_mask)[0]
                    thinking_vectors = hidden_states[batch_idx, thinking_indices]

                    refined_vectors = self.refiner(thinking_vectors.to(dtype=self.dtype))
                    final_vectors = thinking_vectors + self.alpha * refined_vectors
                    final_vectors = self.norm(final_vectors)
                    final_vectors = final_vectors.to(dtype=hidden_states.dtype)

                    hidden_states[batch_idx, thinking_indices] = final_vectors

        return hidden_states


class CodebookAttention(nn.Module):
    """Codebook-based thinking module using Query->Codebook cross-attention"""

    def __init__(self, model, codebook_size: int, select_len: int):
        assert not isinstance(model, CodebookAttention)
        super().__init__()
        self.model = model
        self.codebook_size = codebook_size
        self.select_len = select_len

        device = next(model.parameters()).device

        if hasattr(self.model, 'config') and hasattr(self.model.config, 'hidden_size'):
            self.hidden_size = self.model.config.hidden_size
        elif hasattr(self.model, 'hidden_size'):
            self.hidden_size = self.model.hidden_size
        else:
            for param in self.model.parameters():
                if len(param.shape) >= 2:
                    self.hidden_size = param.shape[-1]
                    break
            else:
                self.hidden_size = 4096

        DROPOUT_RATE = 0.1
        model_dtype = next(model.parameters()).dtype

        self.codebook = nn.Parameter(
            torch.randn(self.codebook_size, self.hidden_size, device=device, dtype=model_dtype)
        )

        self.learnable_queries = nn.Parameter(
            torch.randn(self.select_len, self.hidden_size, device=device, dtype=model_dtype)
        )

        self.question_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=DROPOUT_RATE,
            batch_first=True,
            device=device,
            dtype=model_dtype
        )

        self.question_semantic_extractor = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size, dtype=model_dtype),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(self.hidden_size, self.hidden_size, dtype=model_dtype)
        ).to(device)

        self.codebook_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=DROPOUT_RATE,
            batch_first=True,
            device=device,
            dtype=model_dtype
        )

        self.thinking_norm = RMSNorm(self.hidden_size).to(device=device, dtype=model_dtype)

        nn.init.normal_(self.learnable_queries, std=0.02)
        nn.init.normal_(self.codebook, std=0.02)

        self.to(device=device, dtype=model_dtype)
    
    def _ensure_device_consistency(self, target_device, target_dtype):
        if self.codebook.device != target_device or self.codebook.dtype != target_dtype:
            self.codebook.data = self.codebook.data.to(device=target_device, dtype=target_dtype)

        if self.learnable_queries.device != target_device or self.learnable_queries.dtype != target_dtype:
            self.learnable_queries.data = self.learnable_queries.data.to(device=target_device, dtype=target_dtype)

        self.question_semantic_extractor = self.question_semantic_extractor.to(device=target_device, dtype=target_dtype)
        self.question_attention = self.question_attention.to(device=target_device, dtype=target_dtype)
        self.codebook_attention = self.codebook_attention.to(device=target_device, dtype=target_dtype)
        self.thinking_norm = self.thinking_norm.to(device=target_device, dtype=target_dtype)

    def forward(self, **kwargs):
        hidden_states = kwargs['hidden_states']
        input_mask = kwargs['input_mask']
        full_attention_mask = kwargs['full_attention_mask']
        past_key_value = kwargs.get('past_key_value', None)

        bsz, seq_len, hidden_size = hidden_states.size()

        is_prefill = (seq_len == input_mask.size(1))
        should_process = is_prefill and (input_mask == 2).any()

        adapter = getattr(self, '_adapter_ref', lambda: None)()
        adapter_ctx = adapter._runtime_ctx if adapter else None

        if should_process and adapter_ctx is not None:
            adapter_ctx["inject_cnt"] = adapter_ctx.get("inject_cnt", 0) + 1

        if should_process:
            device = hidden_states.device
            dtype = hidden_states.dtype
            self._ensure_device_consistency(device, dtype)

            question_mask = input_mask == 1
            thinking_mask = input_mask == 2

            question_embeddings = self._extract_question_embeddings(hidden_states, question_mask)
            adjusted_queries = self._adjust_queries_with_question(question_embeddings)

            batch_codebook = self.codebook.to(device=device, dtype=dtype).unsqueeze(0).expand(bsz, -1, -1)

            thinking_vectors, attention_weights = self.codebook_attention(
                query=adjusted_queries,
                key=batch_codebook,
                value=batch_codebook
            )

            thinking_vectors = self.thinking_norm(thinking_vectors)
            self._inject_thinking_vectors(hidden_states, thinking_vectors, thinking_mask)

        kwargs['hidden_states'] = hidden_states
        attn_output, attn_weights = self.model(**kwargs)

        return attn_output, attn_weights

    def _extract_question_embeddings(self, hidden_states, question_mask):
        batch_size = hidden_states.size(0)
        device = hidden_states.device
        dtype = hidden_states.dtype

        question_embeddings = []
        max_question_len = 0

        for bs in range(batch_size):
            question_positions = question_mask[bs]
            if question_positions.sum() > 0:
                seq_len = hidden_states.size(1)
                if question_positions.size(0) > seq_len:
                    question_positions = question_positions[:seq_len]

                try:
                    question_seq = hidden_states[bs][question_positions]
                    if question_seq.size(0) > 0:
                        question_embeddings.append(question_seq)
                        max_question_len = max(max_question_len, question_seq.size(0))
                    else:
                        question_embeddings.append(torch.zeros(1, hidden_states.size(-1),
                                                              device=device, dtype=dtype))
                        max_question_len = max(max_question_len, 1)
                except Exception as e:
                    question_embeddings.append(torch.zeros(1, hidden_states.size(-1),
                                                          device=device, dtype=dtype))
                    max_question_len = max(max_question_len, 1)
            else:
                question_embeddings.append(torch.zeros(1, hidden_states.size(-1),
                                                      device=device, dtype=dtype))
                max_question_len = max(max_question_len, 1)

        padded_questions = []
        for question_seq in question_embeddings:
            if question_seq.size(0) < max_question_len:
                padding = torch.zeros(max_question_len - question_seq.size(0), question_seq.size(1),
                                    device=device, dtype=dtype)
                question_seq = torch.cat([question_seq, padding], dim=0)
            padded_questions.append(question_seq)

        return torch.stack(padded_questions, dim=0)

    def _adjust_queries_with_question(self, question_embeddings):
        """Adjust query vectors based on question context using cross-attention"""
        bsz = question_embeddings.size(0)
        device = question_embeddings.device
        dtype = question_embeddings.dtype

        clean_question_embeddings = self.question_semantic_extractor(question_embeddings)

        base_queries = self.learnable_queries.unsqueeze(0).expand(bsz, -1, -1).to(
            device=device, dtype=dtype
        )

        question_aware_queries, attention_weights = self.question_attention(
            query=base_queries,
            key=clean_question_embeddings,
            value=clean_question_embeddings
        )

        final_queries = base_queries + 0.3 * (question_aware_queries - base_queries)

        return final_queries

    def _inject_thinking_vectors(self, hidden_states, thinking_vectors, thinking_mask):
        """Inject generated thinking vectors into specified positions"""
        bsz = hidden_states.size(0)

        thinking_vectors = thinking_vectors.to(device=hidden_states.device, dtype=hidden_states.dtype)

        for bs in range(bsz):
            thinking_positions = thinking_mask[bs]

            if thinking_positions.sum() > 0:
                thinking_indices = torch.where(thinking_positions)[0]

                seq_len = hidden_states.size(1)
                valid_mask = thinking_indices < seq_len
                thinking_indices = thinking_indices[valid_mask]

                if len(thinking_indices) == 0:
                    continue

                num_thinking_positions = len(thinking_indices)
                if num_thinking_positions != self.select_len:
                    num_to_inject = min(num_thinking_positions, self.select_len)
                    thinking_indices = thinking_indices[:num_to_inject]
                    inject_vectors = thinking_vectors[bs][:num_to_inject]
                else:
                    inject_vectors = thinking_vectors[bs]

                if thinking_indices.max().item() >= seq_len or thinking_indices.min().item() < 0:
                    continue

                hidden_states[bs, thinking_indices] = inject_vectors



