"""Simplified router model with minimal feature extraction and direct delta prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

from trainer.encoder import QwenEncoder


def extract_question_vector(tokenizer, codebook_model, instruction, question, max_len=2048, device="cuda"):
    """Extract question vector using codebook model (unified with R_non extraction).
    
    Args:
        tokenizer: Tokenizer for encoding
        codebook_model: Codebook model (same as used for R_non extraction)
        instruction: Instruction text
        question: Question text
        max_len: Maximum sequence length
        device: Device to use
        
    Returns:
        Question vector of shape [H]
    """
    # Use QwenEncoder with codebook select_len for consistency
    select_len = codebook_model.peft_config.select_len
    encoder = QwenEncoder(tokenizer, select_len=select_len)
    enc_output = encoder.encode_inference(
        instruction, 
        "\n[Question]: " + question
    )[0]  # input_mask==1 marks question tokens
    
    # Prepare tensors
    for k in ("input_ids", "attention_mask", "full_attention_mask", "input_mask"):
        enc_output[k] = torch.tensor(enc_output[k][:max_len]).unsqueeze(0).to(device)
    
    # Forward pass through codebook model (same as R_non extraction)
    with torch.no_grad():
        out = codebook_model(
            input_ids=enc_output["input_ids"],
            attention_mask=enc_output["attention_mask"],
            full_attention_mask=enc_output["full_attention_mask"],
            input_mask=enc_output["input_mask"],
            output_hidden_states=True,
            use_cache=False
        )
    
    # Extract last hidden state
    last_hidden = out.hidden_states[-1][0]  # [seq_len, H]
    
    # Pool over question tokens (input_mask==1)
    q_mask = (enc_output["input_mask"][0] == 1)
    if q_mask.any():
        q_vector = last_hidden[q_mask].mean(dim=0)
    else:
        # Fallback to mean pooling over all tokens
        q_vector = last_hidden.mean(dim=0)
    
    return q_vector  # [H]


def extract_thinking_vectors(tokenizer, enc_model, instruction, question,
                            codebook_model, max_len=2048, device="cuda"):
    """Extract thinking vectors from non-thinking forward pass.

    Args:
        tokenizer: Tokenizer for encoding
        enc_model: Frozen Instruct model
        instruction: Instruction text
        question: Question text
        codebook_model: Model with codebook to extract thinking vectors
        max_len: Maximum sequence length
        device: Device to use

    Returns:
        Thinking vectors of shape [L, H]
    """
    # Use QwenEncoder to get proper masks with thinking positions
    # Get select_len from peft_config (as in evaluate_parallel.py)
    select_len = codebook_model.peft_config.select_len
    encoder = QwenEncoder(tokenizer, select_len=select_len)
    enc_output = encoder.encode_inference(instruction, question)[0]
    
    # Prepare tensors
    for k in ("input_ids", "attention_mask", "full_attention_mask", "input_mask"):
        enc_output[k] = torch.tensor(enc_output[k][:max_len]).unsqueeze(0).to(device)
    
    # Forward pass through model with codebook
    with torch.no_grad():
        out = codebook_model(
            input_ids=enc_output["input_ids"],
            attention_mask=enc_output["attention_mask"],
            full_attention_mask=enc_output["full_attention_mask"],
            input_mask=enc_output["input_mask"],
            output_hidden_states=True,
            use_cache=False
        )
    
    # Extract hidden states at thinking positions (input_mask==2)
    last_hidden = out.hidden_states[-1][0]  # [seq_len, H]
    thinking_mask = (enc_output["input_mask"][0] == 2)

    if thinking_mask.any():
        thinking_vectors = last_hidden[thinking_mask]  # [L, H]
        # Ensure we have exactly select_len vectors
        if thinking_vectors.shape[0] != select_len:
            # Pad or truncate if necessary
            if thinking_vectors.shape[0] < select_len:
                # Pad with zeros
                padding = torch.zeros(select_len - thinking_vectors.shape[0], thinking_vectors.shape[1]).to(device)
                thinking_vectors = torch.cat([thinking_vectors, padding], dim=0)
            else:
                # Truncate
                thinking_vectors = thinking_vectors[:select_len]
    else:
        # No thinking positions found - use zero vectors
        thinking_vectors = torch.zeros(select_len, last_hidden.shape[-1]).to(device)

    return thinking_vectors  # [L, H]


class ProtoEncoderAttn(nn.Module):
    """Attention-based prototype encoder with q-conditioned aggregation."""
    
    def __init__(self, hidden_size):
        super().__init__()
        self.W_p = nn.Linear(hidden_size, hidden_size)
        self.U_q = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, prototypes, q_vec=None, k_top=None):
        """
        Args:
            prototypes: Prototype tokens [L, H]
            q_vec: Question vector [H] for q-conditioned attention (optional)
            k_top: Optional top-k sparsification for attention (default: None)
            
        Returns:
            Tuple of (aggregated representation [H], attention weights [L])
        """
        # Transform prototypes
        W_p = self.W_p(prototypes)  # [L, H]
        
        # Add question conditioning if provided
        if q_vec is not None:
            U_q = self.U_q(q_vec).unsqueeze(0)  # [1, H]
            scores_input = torch.tanh(W_p + U_q)  # [L, H]
        else:
            scores_input = torch.tanh(W_p)  # [L, H]
        
        # Compute attention scores
        e = self.v(scores_input).squeeze(-1)  # [L]
        
        # Optional top-k sparsification for cleaner attention
        if k_top is not None and k_top < e.shape[0]:
            topk = torch.topk(e, k_top).indices
            mask = torch.full_like(e, float('-inf'))
            mask[topk] = 0.0
            e = e + mask
        
        alpha = torch.softmax(e, dim=-1)  # [L]
        
        # Weighted sum
        r = (alpha.unsqueeze(-1) * prototypes).sum(0)  # [H]
        
        return r, alpha


class RouterLite(nn.Module):
    """Streamlined router: minimal viable architecture"""

    def __init__(self, hidden_size, align_dim=128, temp_tau=1.0, bias_max=0.5):
        """
        Args:
            hidden_size: Original hidden layer dimension (H)
            align_dim: Aligned low dimension (d), default 128
            temp_tau: Length feature temperature parameter, default 1.0
            bias_max: Maximum value for threshold bias, default 0.5 (reduced)
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.align_dim = align_dim
        self.temp_tau = temp_tau
        self.bias_max = bias_max

        # Core hyperparameter: threshold θ (FP32)
        self.theta = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        # 1) Alignment tower: Query tower (H → d)
        self.mlp_q = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, align_dim * 2),
            nn.GELU(),
            nn.Linear(align_dim * 2, align_dim)
        )

        # 2) Alignment tower: Thinking tower (H → d)
        self.mlp_r = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, align_dim * 2),
            nn.GELU(),
            nn.Linear(align_dim * 2, align_dim)
        )

        # 3) Main decision head: feature interaction → raw_logit (streamlined to 2d)
        interaction_dim = 2 * align_dim  # [z_q; r̄]
        self.mlp_dec = nn.Sequential(
            nn.LayerNorm(interaction_dim),
            nn.Linear(interaction_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

        # 4) Scalar auxiliary feature linear transformation (FP32)
        # β for [uₙ, len_feat] → δθ (removed Δc̃)
        self.beta = nn.Parameter(torch.zeros(2, dtype=torch.float32))

        # 5) Length regression head (one-shot): [similarity, uₙ] → Δℓ_norm
        self.delta_len_head = nn.Linear(2, 1, bias=True)

        # Initialization
        self._init_weights()

    def _init_weights(self):
        """Weight initialization"""
        with torch.no_grad():
            # Initialize β to small values to prevent excessive threshold bias
            self.beta.data.uniform_(-0.1, 0.1)

            # Initialize length regression head
            self.delta_len_head.weight.data.uniform_(-0.1, 0.1)
            self.delta_len_head.bias.data.zero_()

    def _align_representations(self, q_vec, R_non):
        """
        Streamlined alignment: remove redundant attention projection, simplify interaction vectors

        Args:
            q_vec: Question vector [H]
            R_non: Thinking vector sequence [L, H]

        Returns:
            z_q: Aligned query vector [d]
            r_bar: Aligned aggregated thinking vector [d]
            attention_weights: Attention weights [L]
            interaction_vector: Streamlined interaction vector [2d]
        """
        # 1) Transform to low-dimensional space through alignment towers
        z_q = self.mlp_q(q_vec)  # [H] → [d]
        Z_r = self.mlp_r(R_non)  # [L, H] → [L, d]

        # 2) L2 normalization for improved numerical stability
        z_q = F.normalize(z_q, dim=-1)
        Z_r = F.normalize(Z_r, dim=-1)  # Normalize each position

        # 3) Direct dot-product attention (remove redundant projections)
        # scores = Z_r @ z_q / √d
        scores = torch.matmul(Z_r, z_q) / (self.align_dim ** 0.5)  # [L]
        attention_weights = torch.softmax(scores, dim=-1)  # [L]

        # 4) Weighted aggregation
        r_bar = torch.sum(attention_weights.unsqueeze(-1) * Z_r, dim=0)  # [d]
        r_bar = F.normalize(r_bar, dim=-1)  # Normalize aggregated result

        # 5) Streamlined interaction vector [z_q; r̄] (2d)
        interaction_vector = torch.cat([z_q, r_bar], dim=-1)  # [2d]

        return z_q, r_bar, attention_weights, interaction_vector

    def _compute_scalar_features(self, z_q, r_bar, attention_weights):
        """
        Compute streamlined scalar features: remove v_think_d anchor, keep only effective signals

        Args:
            z_q: Aligned query vector [d]
            r_bar: Aligned aggregated thinking vector [d]
            attention_weights: Attention weights [L]

        Returns:
            scalar_features: [s, uₙ] Scalar feature vector [2]
        """
        # 1) Similarity: s = cos(z_q, r̄)
        cos_q_r = torch.sum(z_q * r_bar)

        # 2) uₙ = H(attention_weights) / log(L) (normalized attention entropy)
        L = attention_weights.numel()
        entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8))
        u_n = entropy / torch.log(torch.tensor(L, device=attention_weights.device, dtype=attention_weights.dtype))

        # Construct scalar feature vector
        scalar_features = torch.stack([cos_q_r, u_n])

        return scalar_features

    def forward(self, q, R_non):
        """
        Streamlined forward pass: one-shot length regression, unified FP32 precision

        Args:
            q: Question vector [B, H] or [H]
            R_non: Thinking vector sequence [L, H]

        Returns:
            raw_logit: Basic decision logit [B] or scalar
            delta_theta: Threshold bias [B] or scalar
            features: Detailed feature dictionary
        """
        # Handle batch and single input
        if q.dim() == 1:
            q = q.unsqueeze(0)  # [1, H]
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = q.shape[0]
        raw_logits = []
        delta_thetas = []
        all_features = []

        for i in range(batch_size):
            # 1) Upstream alignment and interaction
            z_q, r_bar, attention_weights, interaction_vector = self._align_representations(q[i], R_non)

            # 2) Main decision: feature interaction → raw_logit
            raw_logit = self.mlp_dec(interaction_vector).squeeze()

            # 3) Compute basic scalar features [s, uₙ]
            scalar_features = self._compute_scalar_features(z_q, r_bar, attention_weights)
            s = scalar_features[0]  # Similarity
            u_n = scalar_features[1]  # Attention uncertainty

            # 4) One-shot length regression: [s, uₙ] → Δℓ_norm
            delta_len_norm = self.delta_len_head(scalar_features).squeeze()

            # 5) Compute threshold bias: δθ = β₁·uₙ + β₂·tanh(Δℓ_norm/τ)
            len_feat = torch.tanh(delta_len_norm / self.temp_tau)
            delta_theta = self.beta[0] * u_n + self.beta[1] * len_feat
            delta_theta = torch.clamp(delta_theta, -self.bias_max, self.bias_max)

            raw_logits.append(raw_logit)
            delta_thetas.append(delta_theta)

            # Save detailed features for debugging
            features = {
                'z_q': z_q,
                'r_bar': r_bar,
                'attention_weights': attention_weights,
                'interaction_vector': interaction_vector,
                's': s.item(),
                'u_n': u_n.item(),
                'delta_len_norm': delta_len_norm.item(),
                'len_feat': len_feat.item(),
                'raw_logit': raw_logit.item(),
                'delta_theta': delta_theta.item()
            }
            all_features.append(features)

        # Stack results
        raw_logit = torch.stack(raw_logits)
        delta_theta = torch.stack(delta_thetas)

        if squeeze_output:
            raw_logit = raw_logit.squeeze(0)
            delta_theta = delta_theta.squeeze(0)
            features = all_features[0]
        else:
            features = all_features

        return raw_logit, delta_theta, features

    def compute_scores(self, q, R_non, len_scale=1.0):
        """
        Compute routing decision: unified decision in logit space

        Args:
            q: Question vector [H]
            R_non: Thinking vector sequence [L, H]
            len_scale: Length normalization scale (for compatibility)

        Returns:
            use_think: Whether to use thinking chain
            scores: Detailed score dictionary
        """
        # Get predictions
        raw_logit, delta_theta, features = self.forward(q, R_non)

        # Unified decision: raw_logit >= θ + δθ
        theta_fp32 = self.theta.float()
        decision_threshold = theta_fp32 + delta_theta
        use_think = (raw_logit >= decision_threshold).item()

        # Compute actual length difference (maintain compatibility)
        delta_len_actual = features['delta_len_norm'] * len_scale

        scores = {
            'raw_logit': raw_logit.item(),
            'delta_theta': delta_theta.item(),
            'threshold': theta_fp32.item(),
            'decision_threshold': decision_threshold.item(),
            'p_think': torch.sigmoid(raw_logit - decision_threshold).item(),  # Actual decision probability
            'delta_len': delta_len_actual,
            'delta_len_norm': features['delta_len_norm'],
            's': features['s'],
            'u_n': features['u_n'],
            'len_feat': features['len_feat'],
            'use_think': use_think,
            'margin': (raw_logit - decision_threshold).item()  # Decision boundary distance
        }

        return use_think, scores


def compute_length_statistics(dataset, percentile=95):
    """Compute length statistics for normalization.
    
    Args:
        dataset: RouterDataset with ell_non and ell_think fields
        percentile: Percentile to use (e.g., 95 for p95)
        
    Returns:
        Dictionary with statistics
    """
    all_lengths = []
    for item in dataset.data:
        all_lengths.append(item['ell_non'])
        all_lengths.append(item['ell_think'])
    
    return {
        'mean': np.mean(all_lengths),
        'std': np.std(all_lengths),
        'max': np.max(all_lengths),
        'min': np.min(all_lengths),
        f'p{percentile}': np.percentile(all_lengths, percentile)
    }


def load_router(router_ckpt_path, device="cuda"):
    """Load trained router model.

    Args:
        router_ckpt_path: Path to router checkpoint
        device: Device to load to

    Returns:
        Loaded router model
    """
    checkpoint = torch.load(router_ckpt_path, map_location=device, weights_only=False)

    # Get configuration parameters
    config = checkpoint['config']
    hidden_size = config['hidden_size']

    # Get new architecture parameters if available, otherwise use defaults
    align_dim = config.get('align_dim', 128)
    temp_tau = config.get('temp_tau', 1.0)
    bias_max = config.get('bias_max', 1.0)

    # Initialize model
    model = RouterLite(
        hidden_size=hidden_size,
        align_dim=align_dim,
        temp_tau=temp_tau,
        bias_max=bias_max
    )

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set length scale if available (for backward compatibility)
    if 'len_scale' in checkpoint:
        model.len_scale = checkpoint['len_scale']
    elif 'length_stats' in checkpoint:
        stats = checkpoint['length_stats']
        if 'p95' in stats:
            model.len_scale = stats['p95']
        elif 'max' in stats:
            model.len_scale = stats['max']
    else:
        model.len_scale = 1.0

    return model.to(device)