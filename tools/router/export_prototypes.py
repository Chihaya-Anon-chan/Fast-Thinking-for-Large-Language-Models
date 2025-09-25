"""Export non-thinking prototypes from trained CodebookAdapter."""

import os
import sys
import argparse
import torch
import torch.nn as nn
from pathlib import Path
import transformers

sys.path.append(str(Path(__file__).parent.parent.parent))

from models.codebook import CodebookAdapterModel


def export_prototypes(ckpt_path, base_model_name, output_path, device="cuda"):
    """Export thinking prototypes from CodebookAttention layer.
    
    Args:
        ckpt_path: Path to trained codebook checkpoint
        base_model_name: Base model name (e.g. Qwen/Qwen3-4B-Instruct-2507)
        output_path: Path to save prototypes tensor
        device: Device to use for computation
    
    Returns:
        Prototypes tensor of shape [L, H] where L=select_len, H=hidden_size
    """
    # Load tokenizer
    print(f"Loading tokenizer: {base_model_name}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    print(f"Loading base model: {base_model_name}")
    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model_name, 
        device_map=device, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    # Load CodebookAdapter checkpoint
    print(f"Loading checkpoint from: {ckpt_path}")
    model = CodebookAdapterModel.from_pretrained(
        base_model, 
        ckpt_path
    ).to(device)
    model.eval()
    
    # Access CodebookAttention layer at the inserted layer
    # The CodebookAttention is wrapped as self_attn at the insertion layer
    inserted_layer = model.peft_config.inserted_layer
    codebook_layer = None
    layer_name = None
    
    for name, module in model.named_modules():
        # CodebookAttention replaces self_attn at the insertion layer
        if name == f"model.model.layers.{inserted_layer - 1}.self_attn":
            if hasattr(module, 'codebook') and hasattr(module, 'learnable_queries'):
                codebook_layer = module
                layer_name = name
                break
    
    if codebook_layer is None:
        # Try alternative naming patterns or check by class type
        from models.codebook.layer import CodebookAttention as CodebookAttentionClass
        for name, module in model.named_modules():
            if isinstance(module, CodebookAttentionClass) or module.__class__.__name__ == 'CodebookAttention':
                codebook_layer = module
                layer_name = name
                break
    
    if codebook_layer is None:
        raise ValueError(f"CodebookAttention layer not found in model at layer {inserted_layer}")
    
    print(f"Found CodebookAttention at: {layer_name}")
    
    # Extract learnable queries and codebook
    learnable_queries = codebook_layer.learnable_queries  # [1, L, H]
    codebook = codebook_layer.codebook  # [codebook_size, H]
    
    # Perform cross-attention: Q -> (K, V)
    # This mimics what happens during inference but without input conditioning
    with torch.no_grad():
        # Prepare Q, K, V
        Q = learnable_queries  # [1, L, H]
        K = codebook.unsqueeze(0)  # [1, codebook_size, H]
        V = codebook.unsqueeze(0)  # [1, codebook_size, H]
        
        # Apply query/key projections if they exist
        if hasattr(codebook_layer, 'q_proj'):
            Q = codebook_layer.q_proj(Q)
        if hasattr(codebook_layer, 'k_proj'):
            K = codebook_layer.k_proj(K)
        if hasattr(codebook_layer, 'v_proj'):
            V = codebook_layer.v_proj(V)
        
        # Compute attention scores
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)  # [1, L, codebook_size]
        
        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        thinking_tokens = torch.matmul(attn_weights, V)  # [1, L, H]

        # Apply thinking norm if exists
        if hasattr(codebook_layer, 'thinking_norm'):
            thinking_tokens = codebook_layer.thinking_norm(thinking_tokens)

        # Extract prototypes [L, H]
        prototypes = thinking_tokens.squeeze(0)  # Remove batch dimension
    
    # [NEW] Numerical stability cleanup
    prototypes = prototypes.float()  # Convert to fp32 for stable operations
    
    # Apply L2 normalization to each prototype vector to suppress extreme energies
    prototypes = torch.nn.functional.normalize(prototypes, dim=-1)  # Each token vector -> L2 normalized
    
    # Optional: Apply small variance scaling to prevent vanishing/exploding gradients
    # prototypes = prototypes * 0.02  # Scale to small variance (optional)
    
    # Print statistics before and after cleanup
    print(f"Original prototype norm mean: {thinking_tokens.squeeze(0).norm(dim=-1).mean().item():.4f}")
    print(f"Cleaned prototype norm mean: {prototypes.norm(dim=-1).mean().item():.4f}")
    print(f"Cleaned prototype norm std: {prototypes.norm(dim=-1).std().item():.4f}")
    
    # Save prototypes with metadata
    L, H = prototypes.shape
    prototype_data = {
        "P_non": prototypes.cpu(),
        "meta": {
            "L": L,
            "H": H,
            "normalized": True,
            "export_method": "cross_attention_with_l2_norm"
        }
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(prototype_data, output_path)
    
    print(f"Exported prototypes shape: {prototypes.shape}")
    print(f"Saved to: {output_path}")
    
    return prototypes


def main():
    parser = argparse.ArgumentParser(description="Export non-thinking prototypes from CodebookAdapter")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to trained codebook checkpoint")
    parser.add_argument("--base", type=str, default="Qwen/Qwen3-4B-Instruct-2507",
                        help="Base model name")
    parser.add_argument("--out", type=str, default="protos/nonthink.pt",
                        help="Output path for prototypes")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    export_prototypes(
        ckpt_path=args.ckpt,
        base_model_name=args.base,
        output_path=args.out,
        device=args.device
    )


if __name__ == "__main__":
    main()