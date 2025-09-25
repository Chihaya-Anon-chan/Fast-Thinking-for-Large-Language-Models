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
import inspect

import torch
import torch.nn as nn
import math

def is_codebook_trainable(params: str) -> bool:
    return any([
        "lora_" in params,
        "learnable_queries" in params,
        "codebook" in params,
        "query_adaptor" in params,
        "question_semantic_extractor" in params,
        "question_attention" in params,
        "codebook_attention" in params,
        "query_attention" in params,
        "thinking_norm" in params,
        "thinking_refiner" in params,
    ])


def hyperfanin_init_weight(linear_layer, hypernet_in, mainnet_in):
    bound = 1 * math.sqrt(3 / (hypernet_in * mainnet_in))
    nn.init.uniform_(linear_layer.weight, -bound, bound)
    nn.init.constant_(linear_layer.bias, 0.0)

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def disable_early_lora(model: torch.nn.Module, inserted_layer: int):
    """Disable LoRA adapters for layers before the inserted layer"""

    for name, module in model.named_modules():
        if not hasattr(module, "disable_adapter"):
            continue

        if ".layers." not in name:
            continue

        try:
            layer_idx = int(name.split(".layers.")[1].split(".")[0]) + 1
        except ValueError:
            continue

        if layer_idx < inserted_layer:
            module.disable_adapter()

def disable_all_lora(model: torch.nn.Module):
    """Disable all LoRA adapters in the model"""
    for _, module in model.named_modules():
        if hasattr(module, "disable_adapter"):
            module.disable_adapter()