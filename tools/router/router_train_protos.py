"""Train router model with learnable thinking prototypes."""

import os
import sys
import json
import jsonlines
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Set tokenizers parallelism to false before importing transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import transformers

sys.path.append(str(Path(__file__).parent.parent.parent))

from tools.router.router_model import (
    RouterLite,
    extract_question_vector,
    extract_thinking_vectors,
    compute_length_statistics
)


class RouterDataset(Dataset):
    """Dataset for router training with stratified validation split and optimized preprocessing."""
    
    def __init__(self, data_path, tokenizer=None, base_model=None, codebook_model=None, device="cuda", 
                 precompute_vectors=True, max_len=2048, split='train', val_ratio=0.15, random_seed=42):
        """Load router training data with stratified validation split.
        
        Args:
            data_path: Path to training data
            tokenizer: Tokenizer for encoding (if precompute_vectors=True)
            base_model: Base model for question encoding (if precompute_vectors=True)
            codebook_model: Model with codebook for thinking extraction (if precompute_vectors=True)
            device: Device for computation
            precompute_vectors: Whether to precompute and cache question vectors
            max_len: Maximum sequence length for encoding
            split: 'train', 'val', or 'all' - which split to return
            val_ratio: Fraction of data to use for validation (0.0-1.0)
            random_seed: Random seed for reproducible splits
        """
        self.precompute_vectors = precompute_vectors
        self.question_vectors = []
        self.thinking_vectors = []
        
        # Load all data first
        all_data = []
        with jsonlines.open(data_path) as reader:
            for obj in reader:
                all_data.append(obj)
        
        
        # Perform stratified split based on sample categories
        if val_ratio > 0.0 and split in ['train', 'val']:
            train_data, val_data, split_stats = self._stratified_split(all_data, val_ratio, random_seed)
            
            if split == 'train':
                self.data = train_data
            else:  # split == 'val'
                self.data = val_data
        else:
            # Use all data
            self.data = all_data
        
        # Precompute question and thinking vectors if requested
        if precompute_vectors and tokenizer is not None and base_model is not None:
            self._precompute_vectors(tokenizer, base_model, codebook_model, device, max_len)
    
    def _stratified_split(self, all_data, val_ratio, random_seed):
        """Perform stratified split to maintain category distribution."""
        import random
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Categorize samples
        categories = {
            'only_non': [],
            'only_think': [], 
            'both_correct': [],
            'both_wrong': []
        }
        
        for i, item in enumerate(all_data):
            a_non = float(item['a_non'])
            a_think = float(item['a_think'])
            
            if a_non == 1.0 and a_think == 0.0:
                categories['only_non'].append(i)
            elif a_non == 0.0 and a_think == 1.0:
                categories['only_think'].append(i)
            elif a_non == 1.0 and a_think == 1.0:
                categories['both_correct'].append(i)
            else:  # both wrong
                categories['both_wrong'].append(i)
        
        
        # Stratified sampling within each category
        train_indices = []
        val_indices = []
        split_stats = {}
        
        for category, indices in categories.items():
            if len(indices) == 0:
                split_stats[category] = {'total': 0, 'train': 0, 'val': 0}
                continue
                
            # Shuffle indices for random sampling
            shuffled_indices = indices.copy()
            random.shuffle(shuffled_indices)
            
            # Calculate split point
            val_count = max(1, int(len(indices) * val_ratio))  # At least 1 sample for val
            train_count = len(indices) - val_count
            
            # Ensure we don't take all samples for validation
            if val_count >= len(indices):
                val_count = len(indices) - 1
                train_count = 1
            
            # Split indices
            val_category_indices = shuffled_indices[:val_count]
            train_category_indices = shuffled_indices[val_count:]
            
            train_indices.extend(train_category_indices)
            val_indices.extend(val_category_indices)
            
            split_stats[category] = {
                'total': len(indices),
                'train': len(train_category_indices),
                'val': len(val_category_indices)
            }
        
        # Create split datasets
        train_data = [all_data[i] for i in train_indices]
        val_data = [all_data[i] for i in val_indices]
        
        # Shuffle final datasets
        random.shuffle(train_data)
        random.shuffle(val_data)
        
        return train_data, val_data, split_stats
    
    def _precompute_vectors(self, tokenizer, base_model, codebook_model, device, max_len):
        """Precompute all question and thinking vectors for faster training."""
        
        self.question_vectors = []
        self.thinking_vectors = []
        
        for item in tqdm(self.data, desc="Computing vectors"):
            # Compute question vector using codebook model (unified source)
            q_vec = extract_question_vector(
                tokenizer, codebook_model, 
                item['instruction'], item['question'], 
                max_len=max_len, device=device
            )
            self.question_vectors.append(q_vec.cpu())  # Store on CPU to save GPU memory
            
            # Compute thinking vectors if codebook model provided
            if codebook_model is not None:
                r_vec = extract_thinking_vectors(
                    tokenizer, base_model,
                    item['instruction'], item['question'],
                    codebook_model, max_len=max_len, device=device
                )
                self.thinking_vectors.append(r_vec.cpu())
        
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        result = {
            'instruction': item['instruction'],
            'question': item['question'],
            'a_non': float(item['a_non']),  # accuracy non-thinking
            'a_think': float(item['a_think']),  # accuracy thinking
            'len_non': float(item['ell_non']),  # tokens non-thinking (renamed)
            'len_think': float(item['ell_think']),  # tokens thinking (renamed)
        }
        
        # Include precomputed vectors if available
        if self.precompute_vectors and len(self.question_vectors) > idx:
            result['q_vector'] = self.question_vectors[idx]
        if self.precompute_vectors and len(self.thinking_vectors) > idx:
            result['r_non'] = self.thinking_vectors[idx]
            
        return result
    
    def normalize_lengths(self):
        """Compute normalization statistics for token lengths."""
        all_lengths = []
        for item in self.data:
            all_lengths.append(item['ell_non'])
            all_lengths.append(item['ell_think'])
        
        return {
            'mean': np.mean(all_lengths),
            'std': np.std(all_lengths),
            'max': np.max(all_lengths),
            'min': np.min(all_lengths)
        }
    
    def get_category_stats(self):
        """Get statistics about sample categories in current split."""
        categories = {'only_non': 0, 'only_think': 0, 'both_correct': 0, 'both_wrong': 0}
        
        for item in self.data:
            a_non = float(item['a_non'])
            a_think = float(item['a_think'])
            
            if a_non == 1.0 and a_think == 0.0:
                categories['only_non'] += 1
            elif a_non == 0.0 and a_think == 1.0:
                categories['only_think'] += 1
            elif a_non == 1.0 and a_think == 1.0:
                categories['both_correct'] += 1
            else:
                categories['both_wrong'] += 1
                
        return categories


def collate_fn(batch):
    """Custom collate function for router dataset with precomputed vectors."""
    result = {
        'instruction': [item['instruction'] for item in batch],
        'question': [item['question'] for item in batch],
        'a_non': torch.tensor([item['a_non'] for item in batch]),
        'a_think': torch.tensor([item['a_think'] for item in batch]),
        'len_non': torch.tensor([item['len_non'] for item in batch]),
        'len_think': torch.tensor([item['len_think'] for item in batch]),
    }
    
    # Include precomputed vectors if available
    if 'q_vector' in batch[0]:
        result['q_vectors'] = torch.stack([item['q_vector'] for item in batch])
    if 'r_non' in batch[0]:
        result['r_non_batch'] = torch.stack([item['r_non'] for item in batch])
    
    return result


def compute_loss_refactored(
    model_output, targets, *,
    len_scale=1.0,
    beta_reg=1e-4,               # L2 regularization weight to prevent excessive δθ
    w_len=0.1                    # Length regression loss weight (reduced, no longer dominant)
):
    """Reconstructed loss function: unified BCEWithLogits(raw_logit - (θ + δθ), y)

    Args:
        model_output: Dictionary containing raw_logit, delta_theta, delta_len
        targets: Dictionary containing a_non, a_think, len_non, len_think
        len_scale: Length normalization scale
        beta_reg: L2 regularization weight for β parameter
        w_len: Length regression loss weight

    Returns:
        total_loss, loss_dict
    """
    device = targets['a_non'].device

    # Predictions
    raw_logit = model_output['raw_logit'].float()
    delta_theta = model_output['delta_theta'].float()
    delta_len_norm = model_output['delta_len'].float()
    theta = model_output.get('theta', torch.tensor(0.0, device=device, dtype=torch.float32))

    # Targets
    tgt_a_non = targets['a_non'].float()
    tgt_a_think = targets['a_think'].float()
    tgt_len_non = targets['len_non'].to(device).float()
    tgt_len_think = targets['len_think'].to(device).float()

    # Normalized length difference (for regression)
    tgt_delta_len = (tgt_len_think - tgt_len_non) / len_scale

    # Sample classification
    only_non = (tgt_a_non == 1.0) & (tgt_a_think == 0.0)
    only_think = (tgt_a_non == 0.0) & (tgt_a_think == 1.0)
    both_correct = (tgt_a_non == 1.0) & (tgt_a_think == 1.0)
    both_wrong = (tgt_a_non == 0.0) & (tgt_a_think == 0.0)

    # Labels: y=1 for only_think, y=0 for only_non & both_correct
    # Ignore both_wrong
    valid_samples = ~both_wrong
    target = torch.zeros_like(tgt_a_non)
    target[only_think] = 1.0  # Only only_think needs thinking chain

    if valid_samples.sum() == 0:
        # All samples are both_wrong, return zero loss
        routing_loss = torch.tensor(0.0, device=device)
        len_loss = torch.tensor(0.0, device=device)
        reg_loss = torch.tensor(0.0, device=device)
        pos_weight = 1.0
        n_pos = 0
        n_neg = 0
    else:
        # Core loss: BCEWithLogits(raw_logit - (θ + δθ), y)
        # Unified decision formula: raw_logit ≥ θ + δθ → raw_logit - (θ + δθ) ≥ 0
        decision_logit = raw_logit - (theta + delta_theta)

        # Automatic class balancing: calculate positive/negative sample ratio for current batch
        valid_target = target[valid_samples]
        n_pos_tensor = valid_target.sum()
        n_neg_tensor = (valid_target == 0).sum()

        # Convert to python numbers for return dict
        n_pos = n_pos_tensor.item()
        n_neg = n_neg_tensor.item()

        if n_pos > 0 and n_neg > 0:
            # Set pos_weight as negative_samples/positive_samples for class balance
            pos_weight_tensor = n_neg_tensor / n_pos_tensor
            # Limit pos_weight range to avoid extreme values causing training instability
            pos_weight_tensor = torch.clamp(pos_weight_tensor, min=0.1, max=10.0)
            pos_weight = pos_weight_tensor.item()
        elif n_pos > 0:  # Only positive samples
            pos_weight_tensor = torch.tensor(1.0, device=device, dtype=torch.float32)
            pos_weight = 1.0
        else:  # Only negative samples or no valid samples
            pos_weight_tensor = torch.tensor(1.0, device=device, dtype=torch.float32)
            pos_weight = 1.0

        # BCE loss (auto-balanced)
        routing_loss = F.binary_cross_entropy_with_logits(
            decision_logit[valid_samples],
            valid_target,
            pos_weight=pos_weight_tensor
        )

        # Length regression loss (lightweight auxiliary)
        len_loss = w_len * F.smooth_l1_loss(delta_len_norm, tgt_delta_len)

        # β regularization: prevent excessive threshold bias
        if 'beta' in model_output:
            beta = model_output['beta']
            reg_loss = beta_reg * torch.sum(beta ** 2)
        else:
            reg_loss = torch.tensor(0.0, device=device)

    total_loss = routing_loss + len_loss + reg_loss

    return total_loss, {
        'routing_loss': routing_loss.detach(),
        'len_loss': len_loss.detach(),
        'reg_loss': reg_loss.detach(),
        'total_loss': total_loss.detach(),
        'pos_weight': pos_weight,
        'n_valid': valid_samples.sum().item(),
        'n_pos': n_pos,
        'n_neg': n_neg
    }




def validate_router(model, val_dataloader, device, len_scale, beta_reg=1e-4, w_len=0.1):
    """Validate the router model."""
    model.eval()

    total_loss = 0.0
    total_routing_loss = 0.0
    total_len_loss = 0.0
    total_samples = 0

    correct_predictions = 0
    total_decisions = 0

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation"):
            # Use model dtype for consistency
            model_dtype = next(model.parameters()).dtype
            q_vectors = batch['q_vectors'].to(device, dtype=model_dtype)
            r_non_batch = batch['r_non_batch'].to(device, dtype=model_dtype)

            targets = {
                'a_non': batch['a_non'].to(device),
                'a_think': batch['a_think'].to(device),
                'len_non': batch['len_non'].to(device),
                'len_think': batch['len_think'].to(device),
            }

            # Forward pass
            raw_logit_list = []
            delta_theta_list = []
            delta_len_list = []

            for i in range(q_vectors.shape[0]):
                # Direct forward pass without autocast (unified FP32)
                raw_logit, delta_theta, features = model(q_vectors[i], r_non_batch[i])
                raw_logit_list.append(raw_logit)
                delta_theta_list.append(delta_theta)

                if isinstance(features, dict) and 'delta_len_norm' in features:
                    delta_len = features['delta_len_norm']
                    if isinstance(delta_len, (int, float)):
                        delta_len = torch.tensor(delta_len, device=raw_logit.device, dtype=raw_logit.dtype)
                    delta_len_list.append(delta_len)
                else:
                    delta_len_list.append(torch.tensor(0.0, device=raw_logit.device, dtype=raw_logit.dtype))

            model_output = {
                'raw_logit': torch.stack(raw_logit_list),
                'delta_theta': torch.stack(delta_theta_list),
                'delta_len': torch.stack(delta_len_list)
            }

            # Add theta and beta for gradient computation and regularization
            if hasattr(model, 'theta'):
                model_output['theta'] = model.theta
            if hasattr(model, 'beta'):
                model_output['beta'] = model.beta

            # Compute loss
            loss, loss_dict = compute_loss_refactored(
                model_output, targets,
                len_scale=len_scale,
                beta_reg=beta_reg,
                w_len=w_len
            )

            # Accumulate metrics
            batch_size = q_vectors.shape[0]
            total_loss += loss.item() * batch_size
            total_routing_loss += loss_dict['routing_loss'].item() * batch_size
            total_len_loss += loss_dict['len_loss'].item() * batch_size
            total_samples += batch_size

            # Compute routing accuracy using unified decision formula
            raw_logits = model_output['raw_logit']
            delta_thetas = model_output['delta_theta']
            theta = model_output.get('theta', torch.tensor(0.0, device=raw_logits.device, dtype=torch.float32))
            # Unified decision formula: raw_logit ≥ θ + δθ
            decision_logits = raw_logits - (theta + delta_thetas)
            predictions = (decision_logits >= 0).float()

            # True labels
            tgt_a_non = targets['a_non']
            tgt_a_think = targets['a_think']
            only_think = (tgt_a_non == 0.0) & (tgt_a_think == 1.0)
            ground_truth = only_think.float()

            # Only count non-both_wrong samples
            both_wrong = (tgt_a_non == 0.0) & (tgt_a_think == 0.0)
            valid_mask = ~both_wrong

            if valid_mask.sum() > 0:
                correct_predictions += (predictions[valid_mask] == ground_truth[valid_mask]).sum().item()
                total_decisions += valid_mask.sum().item()

    # Calculate averages
    avg_loss = total_loss / total_samples
    avg_routing_loss = total_routing_loss / total_samples
    avg_len_loss = total_len_loss / total_samples
    routing_acc = correct_predictions / total_decisions if total_decisions > 0 else 0.0

    return {
        'loss': avg_loss,
        'routing_loss': avg_routing_loss,
        'len_loss': avg_len_loss,
        'routing_acc': routing_acc,
        'total_samples': total_samples,
        'total_decisions': total_decisions
    }


def train_router(args):
    """Main training function."""

    # Fix DataLoader shared memory issues for multi-worker loading
    try:
        import torch.multiprocessing as mp
        mp.set_sharing_strategy('file_system')
    except RuntimeError:
        # Already set or not applicable, continue
        pass

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Enable optimizations for bf16
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    
    # Load base model for question encoding
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map=device,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    base_model.eval()  # Freeze base model
    
    # Get model dimensions
    hidden_size = base_model.config.hidden_size
    
    # Load codebook model for thinking extraction (REQUIRED for RouterLite)
    if not hasattr(args, 'codebook_path') or not args.codebook_path:
        raise ValueError("--codebook_path is required for RouterLite to extract thinking vectors")
    
    # Import and load the codebook model using the same approach as evaluate_parallel.py
    from models.codebook import CodebookAdapterModel
    codebook_model = CodebookAdapterModel.from_pretrained(
        base_model, args.codebook_path
    ).to(device)
    codebook_model.eval()  # Freeze codebook model
    
    # Get select_len from peft_config (as in evaluate_parallel.py)
    select_len = codebook_model.peft_config.select_len
    
    # Load dataset with validation split
    
    # Decide whether to precompute vectors
    precompute = getattr(args, 'precompute_vectors', True)
    val_ratio = getattr(args, 'val_ratio', 0.15)
    
    # Create training dataset
    if precompute:
        train_dataset = RouterDataset(
            args.train,
            tokenizer=tokenizer,
            base_model=base_model,
            codebook_model=codebook_model,
            device=device,
            precompute_vectors=True,
            split='train',
            val_ratio=val_ratio,
            random_seed=getattr(args, 'seed', 42)
        )
    else:
        train_dataset = RouterDataset(
            args.train,
            precompute_vectors=False,
            split='train',
            val_ratio=val_ratio,
            random_seed=getattr(args, 'seed', 42)
        )
    
    # Create validation dataset
    if val_ratio > 0.0:
        if precompute:
            val_dataset = RouterDataset(
                args.train,
                tokenizer=tokenizer,
                base_model=base_model,
                codebook_model=codebook_model,
                device=device,
                precompute_vectors=True,
                split='val',
                val_ratio=val_ratio,
                random_seed=getattr(args, 'seed', 42)
            )
        else:
            val_dataset = RouterDataset(
                args.train,
                precompute_vectors=False,
                split='val',
                val_ratio=val_ratio,
                random_seed=getattr(args, 'seed', 42)
            )

    else:
        val_dataset = None
    
    # Calculate len_scale from training set delta lengths
    if args.len_scale is not None:
        len_scale = float(args.len_scale)
    else:
        # Calculate p95 of |ℓ_think - ℓ_non| (delta lengths) from training set
        delta_lens = np.array([abs(ex["ell_think"] - ex["ell_non"]) for ex in train_dataset.data], dtype=np.float32)
        len_scale = float(np.percentile(delta_lens, 95))

    # Compute length statistics for informational purposes
    length_stats = compute_length_statistics(train_dataset, percentile=95)
    
    # Create optimized dataloaders
    # When precompute_vectors=True, use fewer workers to avoid "Too many open files" error
    # When vectors are precomputed, we don't need many workers since data is cached
    if precompute:
        num_workers = 0  # Avoid shared memory issues with precomputed vectors
    else:
        num_workers = getattr(args, 'num_workers', 6)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.bs,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=False,  # Enable for GPU transfer speed
        persistent_workers=False,  # Disable to avoid memory issues
        prefetch_factor=None  # Disable prefetch
    )
    
    # Create validation dataloader if we have validation set
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.bs,
            shuffle=False,  # No need to shuffle validation set
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=False,  # Enable for GPU transfer speed
            persistent_workers=False,  # Disable to avoid memory issues
            prefetch_factor=None  # Disable prefetch
        )
    else:
        val_dataloader = None
    
    # Initialize RouterLite model (refactored architecture)
    align_dim = getattr(args, 'align_dim', 128)
    temp_tau = getattr(args, 'temp_tau', 1.0)
    bias_max = getattr(args, 'bias_max', 1.0)

    model = RouterLite(
        hidden_size=hidden_size,
        align_dim=align_dim,
        temp_tau=temp_tau,
        bias_max=bias_max
    ).to(device)

    # Use unified FP32 precision to avoid mixed precision issues
    model = model.float()


    # Store length scale as attribute for later use
    model.len_scale = len_scale
    
    # Setup optimizer with different learning rates
    param_groups = []

    # Main model parameters (standard LR)
    main_params = []
    # Critical parameters (higher LR multiplier)
    critical_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if name in ['theta', 'beta']:
                critical_params.append(param)
            else:
                main_params.append(param)

    # Add parameter groups
    if main_params:
        param_groups.append({'params': main_params, 'lr': args.lr})
    if critical_params:
        param_groups.append({
            'params': critical_params,
            'lr': args.lr * args.margin_lr_mult,
            'name': 'critical'
        })

    # Initialize optimizer
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    
    # Learning rate scheduler with warmup
    total_steps = len(train_dataloader) * args.epochs
    warmup_steps = int(getattr(args, 'warmup_ratio', 0.03) * total_steps) if args.warmup_steps is None else args.warmup_steps
    min_lr = getattr(args, 'min_lr', None)
    if min_lr is None:
        min_lr = args.lr * 0.01
    
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps),
            CosineAnnealingLR(optimizer, T_max=(total_steps - warmup_steps), eta_min=min_lr)
        ],
        milestones=[warmup_steps]
    )
    
    # Training loop with best model tracking
    model.train()

    # Best model tracking variables
    best_val_metric = float('-inf')  # We want to maximize routing accuracy
    best_epoch = -1
    best_model_state = None
    save_threshold_epoch = int(args.epochs * args.save_threshold)  # Only save best from last portion
    
    for epoch in range(args.epochs):
        epoch_loss = 0
        epoch_acc_loss = 0
        epoch_len_loss = 0
        
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad(set_to_none=True)
            
            # Get question vectors (either precomputed or computed on-the-fly)
            model_dtype = next(model.parameters()).dtype
            if 'q_vectors' in batch:
                # Use precomputed vectors
                q_vectors = batch['q_vectors'].to(device, dtype=model_dtype)  # [B, H]
            else:
                # Compute vectors on-the-fly using codebook model
                q_vectors = []
                for inst, quest in zip(batch['instruction'], batch['question']):
                    q_vec = extract_question_vector(
                        tokenizer, codebook_model, inst, quest, device=device
                    )
                    q_vectors.append(q_vec)
                q_vectors = torch.stack(q_vectors).to(dtype=model_dtype)  # [B, H]

            # Get thinking vectors (either precomputed or computed on-the-fly)
            if 'r_non_batch' in batch:
                # Use precomputed thinking vectors
                r_non_batch = batch['r_non_batch'].to(device, dtype=model_dtype)  # [B, L, H]
            else:
                # Compute thinking vectors on-the-fly
                r_non_batch = []
                for inst, quest in zip(batch['instruction'], batch['question']):
                    r_non = extract_thinking_vectors(
                        tokenizer, base_model, inst, quest,
                        codebook_model, max_len=2048, device=device
                    )
                    r_non_batch.append(r_non)
                r_non_batch = torch.stack(r_non_batch).to(dtype=model_dtype)  # [B, L, H]
            
            # Move targets to device
            targets = {
                'a_non': batch['a_non'].to(device),
                'a_think': batch['a_think'].to(device),
                'len_non': batch['len_non'].to(device),
                'len_think': batch['len_think'].to(device),
            }
            
            # Simplified forward pass (unified FP32 precision, no autocast)
            raw_logit_list = []
            delta_theta_list = []
            delta_len_list = []
            u_n_list = []

            for i in range(q_vectors.shape[0]):
                raw_logit, delta_theta, features = model(q_vectors[i], r_non_batch[i])
                raw_logit_list.append(raw_logit)
                delta_theta_list.append(delta_theta)

                # Use new field names
                if isinstance(features, dict) and 'delta_len_norm' in features:
                    delta_len = features['delta_len_norm']
                    if isinstance(delta_len, (int, float)):
                        delta_len = torch.tensor(delta_len, device=raw_logit.device, dtype=raw_logit.dtype)
                    delta_len_list.append(delta_len)
                else:
                    delta_len_list.append(torch.tensor(0.0, device=raw_logit.device, dtype=raw_logit.dtype))

                if features is not None and isinstance(features, dict):
                    u_n_val = features.get('u_n', 0.0)
                    if isinstance(u_n_val, (int, float)):
                        u_n_val = torch.tensor(u_n_val, device=raw_logit.device, dtype=raw_logit.dtype)
                    u_n_list.append(u_n_val)

            model_output = {
                'raw_logit': torch.stack(raw_logit_list),
                'delta_theta': torch.stack(delta_theta_list),
                'delta_len': torch.stack(delta_len_list)
            }

            # Add theta and beta for gradient computation and regularization
            if hasattr(model, 'theta'):
                model_output['theta'] = model.theta
            if hasattr(model, 'beta'):
                model_output['beta'] = model.beta

            # Collect u_n for debugging (optional)
            if u_n_list:
                model_output['u_n'] = torch.stack(u_n_list)  # [B]

            # Use reconstructed loss function (unified BCEWithLogits)
            loss, loss_dict = compute_loss_refactored(
                model_output, targets,
                len_scale=len_scale,
                beta_reg=getattr(args, 'beta_reg', 1e-4),
                w_len=getattr(args, 'w_len', 0.1)
            )
            
            # For compatibility with tracking
            acc_loss = loss_dict.get('routing_loss', torch.tensor(0.0))
            len_loss = loss_dict.get('len_loss', torch.tensor(0.0))
            
            # Backward pass with gradient clipping
            loss.backward()



            if getattr(args, 'grad_clip', 1.0) is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            
            optimizer.step()
            scheduler.step()
            
            # Update statistics
            epoch_loss += loss.item()
            epoch_acc_loss += acc_loss.item()
            epoch_len_loss += len_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'route': f'{loss_dict["routing_loss"].item():.3f}',
                'len': f'{loss_dict["len_loss"].item():.3f}',
                'lr': f'{scheduler.get_last_lr()[0]:.1e}'
            })
        
        # Print epoch statistics with margin monitoring
        avg_loss = epoch_loss / len(train_dataloader)
        avg_acc_loss = epoch_acc_loss / len(train_dataloader)
        avg_len_loss = epoch_len_loss / len(train_dataloader)
        
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")

        # Validation
        if val_dataloader is not None and (epoch + 1) % args.eval_every == 0:
            val_metrics = validate_router(model, val_dataloader, device, len_scale, args.beta_reg, args.w_len)

            # Determine validation metric for best model selection
            val_metric = val_metrics.get('routing_acc', val_metrics.get('routing_loss', 0.0))

            # Save best model (only from the last portion of training)
            if epoch >= save_threshold_epoch and val_metric > best_val_metric:
                best_val_metric = val_metric
                best_epoch = epoch
                best_model_state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_metrics': val_metrics
                }
                print(f"New best model at epoch {epoch+1} (acc: {val_metric:.3f})")
        
    
    # Save final model

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'len_scale': len_scale,
        'config': {
            'hidden_size': hidden_size,
            'align_dim': getattr(args, 'align_dim', 128),
            'temp_tau': getattr(args, 'temp_tau', 1.0),
            'bias_max': getattr(args, 'bias_max', 1.0),
        },
        'length_stats': length_stats,
        'train_args': vars(args)
    }

    torch.save(checkpoint, args.out)
    print(f"Training complete. Model saved to {args.out}")


def main():
    # Set multiprocessing start method for CUDA compatibility
    try:
        import multiprocessing as mp
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, ignore
        pass
    
    parser = argparse.ArgumentParser(description="Train router model with prototypes")
    
    # Data arguments
    parser.add_argument("--train", type=str, required=True,
                        help="Path to training data (router_ds.jsonl)")
    parser.add_argument("--p_non", type=str, required=False, default=None,
                        help="Path to non-thinking prototypes (not used in lite mode)")
    parser.add_argument("--codebook_path", type=str, required=True,
                        help="Path to codebook model for thinking extraction (REQUIRED for RouterLite)")
    
    # Model arguments
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-4B-Instruct-2507",
                        help="Base model for question encoding")
    # Removed unused parameters for lite mode
    parser.add_argument("--len_norm", type=str, default="p95", choices=["p95", "max", "none"],
                        help="Length normalization strategy")

    # Refactored architecture parameters
    parser.add_argument("--align_dim", type=int, default=128,
                        help="Alignment dimension for representations (d)")
    parser.add_argument("--temp_tau", type=float, default=1.0,
                        help="Temperature parameter for length features")
    parser.add_argument("--bias_max", type=float, default=1.0,
                        help="Maximum value for threshold bias (δθ)")
    parser.add_argument("--beta_reg", type=float, default=1e-4,
                        help="L2 regularization weight for beta parameters")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--bs", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", "--wd", type=float, default=0.01,
                        help="Weight decay")
    
    parser.add_argument("--w_len", type=float, default=1.0,
                        help="Length regression loss weight")

    parser.add_argument("--margin_lr_mult", type=float, default=5.0,
                        help="Learning rate multiplier for θ/β parameters")
    
    # [NEW] Additional training arguments for improved training
    parser.add_argument("--len_scale", type=float, default=None,
                        help="Override automatic len_scale with fixed value")
    parser.add_argument("--warmup_steps", type=int, default=None,
                        help="Number of warmup steps (default: 3% of total)")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="Warmup ratio of total steps")
    parser.add_argument("--min_lr", type=float, default=None,
                        help="Minimum learning rate for cosine schedule (default: lr * 0.01)")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping max norm")
    
    # Data preprocessing optimizations
    parser.add_argument("--precompute_vectors", action="store_true", default=True,
                        help="Precompute question vectors for faster training")
    parser.add_argument("--num_workers", type=int, default=6,
                        help="Number of dataloader workers (default: 6 for parallel loading)")
    
    # Validation set arguments
    parser.add_argument("--val_ratio", type=float, default=0.15,
                        help="Fraction of data to use for validation (0.0 to disable)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible train/val splits")
    parser.add_argument("--save_best_only", action="store_true", default=True,
                        help="Save only the best model based on validation performance")
    parser.add_argument("--save_threshold", type=float, default=0.5,
                        help="Only save best model from last fraction of epochs (default: 0.5 = last half)")
    parser.add_argument("--eval_every", type=int, default=5,
                        help="Evaluate every N epochs")
    
    # Output arguments
    parser.add_argument("--out", type=str, default="router.pt",
                        help="Output path for trained model")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    
    args = parser.parse_args()
    
    train_router(args)


if __name__ == "__main__":
    main()