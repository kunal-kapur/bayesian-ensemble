import torch
import torch.nn as nn
from typing import Dict

class NISP:
    def __init__(self, model: nn.Module):
        self.model = model
        self.importance_scores = {}
    
    def compute_importance_scores(self, final_layer_scores: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Compute importance scores for all neurons using NISP algorithm
        
        Args:
            final_layer_scores: Importance scores for the final layer neurons.
                              If None, uniform scores will be used.
                              
        Returns:
            Dictionary mapping layer names to their importance scores
        """
        # Get all layers in reverse order (output to input)
        layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                layers.append((name, module))
        layers.reverse()
        
        # Initialize importance scores
        self.importance_scores = {}
        
        # Set final layer scores (use uniform if not provided)
        if final_layer_scores is None:
            final_layer = layers[0][1]
            if isinstance(final_layer, nn.Linear):
                self.importance_scores[layers[0][0]] = torch.ones(final_layer.out_features)
            else:  # Conv2d
                self.importance_scores[layers[0][0]] = torch.ones(final_layer.out_channels)
        else:
            self.importance_scores[layers[0][0]] = final_layer_scores
        
        # Propagate scores backward through the network
        for i in range(1, len(layers)):
            curr_name, curr_module = layers[i]
            prev_name, prev_module = layers[i-1]
            
            # Get weights of the next layer (which connects current to previous)
            weights = prev_module.weight
            
            # For conv layers, we need to handle the spatial dimensions
            if isinstance(prev_module, nn.Conv2d):
                # Flatten the conv weights (out_channels, in_channels, *kernel_size)
                weights = weights.view(weights.size(0), -1)
            
            # Propagate importance scores: s_{k,j} = sum_i |w_{i,j}^{(k+1)}| s_{k+1,i}
            prev_scores = self.importance_scores[prev_name]
            curr_scores = torch.matmul(torch.abs(weights.T), prev_scores.float())
            
            # Store the computed scores
            self.importance_scores[curr_name] = curr_scores
        
        return self.importance_scores
    
    def get_pruning_mask(self, layer_name: str, pruning_rate: float) -> torch.Tensor:
        if layer_name not in self.importance_scores:
            raise ValueError(f"No importance scores computed for layer {layer_name}")
        
        scores = self.importance_scores[layer_name]
        num_neurons = scores.numel()
        
        # Determine how many to keep
        k = max(1, int(num_neurons * (1 - pruning_rate)))  # Keep top (1 - pruning_rate)
        
        # Get top-k threshold
        topk_values, _ = torch.topk(scores, k, largest=True, sorted=False)
        threshold = topk_values.min()
        
        mask = scores >= threshold  # This may still slightly exceed k if tied
        return mask.float()