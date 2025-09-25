"""
Unified Minimal Refiner - configurable minimal refiner for thinking tokens
"""
import torch
import torch.nn as nn

class MinimalThinkingRefiner(nn.Module):
    """Configurable minimal thinking refiner module"""

    def __init__(self, hidden_size: int, dtype=torch.bfloat16, ultra_minimal: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.ultra_minimal = ultra_minimal

        if ultra_minimal:
            # Ultra minimal: only one scalar scaling factor
            self.scale = nn.Parameter(torch.tensor(1.0, dtype=dtype))
        else:
            # Standard minimal: element-wise scaling and offset
            self.scale = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
            self.shift = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))
            self.alpha = nn.Parameter(torch.tensor(0.1, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            input_mask: [batch_size, seq_len], 2 indicates thinking token positions

        Returns:
            refined_hidden_states: [batch_size, seq_len, hidden_size]
        """
        # Only process during complete sequence
        if hidden_states.size(1) != input_mask.size(1):
            return hidden_states

        # Find thinking token positions
        thinking_mask = (input_mask == 2)

        if thinking_mask.any():
            if self.ultra_minimal:
                # Ultra minimal: direct scaling
                hidden_states = hidden_states.clone()
                hidden_states[thinking_mask] = hidden_states[thinking_mask] * self.scale
            else:
                # Standard minimal: per-batch processing with residual connection
                batch_size, seq_len = thinking_mask.shape

                for batch_idx in range(batch_size):
                    batch_thinking_mask = thinking_mask[batch_idx]
                    if batch_thinking_mask.any():
                        thinking_indices = torch.where(batch_thinking_mask)[0]
                        thinking_vectors = hidden_states[batch_idx, thinking_indices]

                        # Apply scaling and offset
                        refined_vectors = thinking_vectors * self.scale + self.shift
                        final_vectors = thinking_vectors + self.alpha * refined_vectors

                        hidden_states[batch_idx, thinking_indices] = final_vectors.to(dtype=hidden_states.dtype)

        return hidden_states


# Backward compatibility aliases
UltraMinimalThinkingRefiner = lambda hidden_size, dtype=torch.bfloat16: MinimalThinkingRefiner(hidden_size, dtype, ultra_minimal=True)

# Backward compatibility aliases
MinimalReflectionRefiner = MinimalThinkingRefiner
UltraMinimalReflectionRefiner = UltraMinimalThinkingRefiner