"""
Learning rate scheduler utilities for per-component learning rate control
"""
import json
import re
from typing import Dict, List, Tuple, Optional
import torch
from torch.optim import Optimizer

class ComponentLRScheduler:
    """
    Manage learning rate multipliers for different components
    """
    
    def __init__(self, config_path: str = "lr_multipliers.json"):
        """
        Initialize learning rate scheduler

        Args:
            config_path: path to learning rate multiplier config file
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.default_multiplier = self.config.get('default_multiplier', 1.0)
    
    def get_parameter_groups(self, model, base_lr: float, stage: str = "stage1", 
                           inserted_layer: Optional[int] = None) -> List[Dict]:
        """
        Create parameter groups based on config, each group with different learning rates

        Args:
            model: model instance
            base_lr: base learning rate
            stage: "stage1" or "stage2"
            inserted_layer: inserted layer position (for conditional logic)

        Returns:
            list of parameter groups, each containing params and lr
        """
        stage_key = f"{stage}_multipliers"
        if stage_key not in self.config:
            print(f"Warning: {stage} config not found, using default learning rate")
            return [{'params': model.parameters(), 'lr': base_lr}]
        
        stage_config = self.config[stage_key]
        components = stage_config.get('components', {})
        
        # Collect all parameters and group them
        param_groups = []
        assigned_params = set()
        
        # Process each component according to config order
        for comp_name, comp_config in components.items():
            multiplier = comp_config.get('multiplier', 1.0)
            patterns = comp_config.get('patterns', [])
            conditional = comp_config.get('conditional', None)
            
            # Collect matching parameters
            matching_params = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                    
                # Check if already assigned
                if id(param) in assigned_params:
                    continue
                
                # Check if matches patterns
                if self._matches_patterns(name, patterns):
                    # Check conditions (if any)
                    if conditional and inserted_layer is not None:
                        layer_num = self._extract_layer_num(name)
                        if layer_num is not None:
                            # Evaluate condition
                            if not self._eval_condition(conditional, layer_num, inserted_layer):
                                continue
                    
                    matching_params.append(param)
                    assigned_params.add(id(param))
            
            # If there are matching parameters, create parameter group
            if matching_params:
                param_groups.append({
                    'params': matching_params,
                    'lr': base_lr * multiplier,
                    'name': comp_name,
                    'multiplier': multiplier
                })
                print(f"  {comp_name}: {len(matching_params)} params, lr={base_lr * multiplier:.2e} (×{multiplier})")
        
        # Process unassigned parameters (use default multiplier)
        unassigned_params = []
        for name, param in model.named_parameters():
            if param.requires_grad and id(param) not in assigned_params:
                unassigned_params.append(param)
        
        if unassigned_params:
            param_groups.append({
                'params': unassigned_params,
                'lr': base_lr * self.default_multiplier,
                'name': 'default',
                'multiplier': self.default_multiplier
            })
            print(f"  default: {len(unassigned_params)} params, lr={base_lr * self.default_multiplier:.2e} (×{self.default_multiplier})")
        
        return param_groups
    
    def _matches_patterns(self, param_name: str, patterns: List[str]) -> bool:
        """
        Check if parameter name matches pattern list

        Supports:
        - Positive match: "codebook"
        - Negative match: "!learnable_queries" (exclude)
        """
        if not patterns:
            return False
        
        matched = False
        for pattern in patterns:
            if pattern.startswith('!'):
                # Negative match
                if pattern[1:] in param_name:
                    return False
            else:
                # Positive match
                if pattern in param_name:
                    matched = True
        
        return matched
    
    def _extract_layer_num(self, param_name: str) -> Optional[int]:
        """
        Extract layer number from parameter name
        """
        match = re.search(r'layers?\.(\d+)', param_name)
        if match:
            return int(match.group(1))
        return None
    
    def _eval_condition(self, condition: str, layer_num: int, inserted_layer: int) -> bool:
        """
        Evaluate condition expression

        Args:
            condition: condition string, e.g. "layer_num < inserted_layer"
            layer_num: current layer number
            inserted_layer: inserted layer position
        """
        try:
            # Create safe evaluation environment
            env = {
                'layer_num': layer_num,
                'inserted_layer': inserted_layer
            }
            return eval(condition, {"__builtins__": {}}, env)
        except:
            return True  # Default include when condition evaluation fails

def create_optimizer_with_lr_scaling(model, base_lr: float, stage: str, 
                                    inserted_layer: Optional[int] = None,
                                    config_path: str = "lr_multipliers.json",
                                    weight_decay: float = 0.01,
                                    optimizer_type: str = "adamw") -> Optimizer:
    """
    Create optimizer with learning rate scaling

    Args:
        model: model instance
        base_lr: base learning rate
        stage: "stage1" or "stage2"
        inserted_layer: inserted layer position
        config_path: config file path
        weight_decay: weight decay
        optimizer_type: optimizer type

    Returns:
        configured optimizer
    """
    scheduler = ComponentLRScheduler(config_path)
    
    print(f"\nConfiguring {stage} learning rate multipliers:")
    param_groups = scheduler.get_parameter_groups(model, base_lr, stage, inserted_layer)
    
    # Add weight_decay to each group
    for group in param_groups:
        group['weight_decay'] = weight_decay
    
    # Create optimizer
    if optimizer_type.lower() == "adamw":
        optimizer = torch.optim.AdamW(param_groups)
    elif optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam(param_groups)
    elif optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(param_groups, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    print(f"Created {optimizer_type} optimizer with {len(param_groups)} parameter groups")
    
    return optimizer

def apply_lr_multiplier_preset(config_path: str, preset: str = "balanced") -> Dict:
    """
    Apply preset learning rate multiplier schemes

    Args:
        config_path: config file path
        preset: preset name ("aggressive", "conservative", "balanced")

    Returns:
        adjusted config
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    if 'experimental_presets' not in config:
        return config
    
    presets = config['experimental_presets']
    if preset not in presets:
        print(f"Warning: Preset {preset} not found")
        return config
    
    preset_config = presets[preset]
    stage1_scale = preset_config.get('stage1_scale', 1.0)
    stage2_scale = preset_config.get('stage2_scale', 1.0)
    
    # Apply scaling
    for comp_config in config['stage1_multipliers']['components'].values():
        comp_config['multiplier'] *= stage1_scale
    
    for comp_config in config['stage2_multipliers']['components'].values():
        comp_config['multiplier'] *= stage2_scale
    
    print(f"Applied preset '{preset}': stage1×{stage1_scale}, stage2×{stage2_scale}")
    
    return config