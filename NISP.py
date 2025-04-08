import torch
import torch.nn as nn
from typing import Dict
from model import ConsistentMCDropout  # Assuming this is a custom module you have


class NISP:
    def __init__(self, model: nn.Module):
        self.model = model
        self.importance_scores = {}

    def compute_aggregated_importance_scores(self, dropout_layers: Dict[str, ConsistentMCDropout],
                                             final_layer_scores: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Compute NISP scores averaged across dropout masks.

        Args:
            dropout_layers: dict of layer name → dropout layer module (with mask_dict)
            final_layer_scores: optional tensor for final layer importance

        Returns:
            Dict[layer_name → averaged importance scores]
        """
        aggregated_scores = {}

        # Assume all masks have the same keys (same number of masks)
        mask_keys = list(next(iter(dropout_layers.values())).mask_dict.keys())

        for mask_id in mask_keys:
            scores = self.compute_importance_scores_for_mask(
                dropout_layers=dropout_layers,
                mask_id=mask_id,
                final_layer_scores=final_layer_scores
            )

            for layer_name, score in scores.items():
                if layer_name not in aggregated_scores:
                    aggregated_scores[layer_name] = []
                aggregated_scores[layer_name].append(score)

        # Aggregate (e.g., mean)
        self.importance_scores = {
            layer: torch.stack(scores).mean(dim=0) for layer, scores in aggregated_scores.items()
        }

        return self.importance_scores

    def compute_importance_scores_for_mask(self, dropout_layers: Dict[str, ConsistentMCDropout],
                                           mask_id: int,
                                           final_layer_scores: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Compute importance scores for a single dropout mask.

        Args:
            dropout_layers: dict mapping layer names to dropout modules
            mask_id: which mask index to use
            final_layer_scores: optional tensor for final layer importance

        Returns:
            Dict[layer_name → importance scores]
        """
        # Collect and reverse layer order
        layers = [(name, module) for name, module in self.model.named_modules()
                  if isinstance(module, (nn.Linear, nn.Conv2d))]
        layers.reverse()

        importance_scores = {}

        # Initialize with uniform score if not provided
        if final_layer_scores is None:
            final_layer = layers[0][1]
            if isinstance(final_layer, nn.Linear):
                importance_scores[layers[0][0]] = torch.ones(final_layer.out_features)
            else:
                importance_scores[layers[0][0]] = torch.ones(final_layer.out_channels)
        else:
            importance_scores[layers[0][0]] = final_layer_scores

        for i in range(1, len(layers)):
            curr_name, curr_module = layers[i]
            prev_name, prev_module = layers[i - 1]

            # Get dropout mask (or all-ones if not found)
            if prev_name in dropout_layers:
                mask = dropout_layers[prev_name].mask_dict[mask_id].float()
            else:
                mask = torch.ones_like(importance_scores[prev_name])

            weights = prev_module.weight
            if isinstance(prev_module, nn.Conv2d):
                weights = weights.view(weights.size(0), -1)

            prev_scores = importance_scores[prev_name]
            curr_scores = torch.matmul(torch.abs(weights.T), prev_scores * mask)

            importance_scores[curr_name] = curr_scores

        return importance_scores
    
    def get_pruning_mask(self, scores: torch.Tensor, pruning_rate: float) -> torch.Tensor:
        """
        Create binary mask to keep top-k neurons based on importance scores.

        Args:
            scores (torch.Tensor): Importance scores for a layer.
            pruning_rate (float): Fraction of neurons to prune (between 0 and 1).

        Returns:
            torch.Tensor: Binary mask with 1s for kept neurons.
        """
        k = int(scores.numel() * (1 - pruning_rate))
        if k <= 0:
            return torch.zeros_like(scores, dtype=torch.bool)
        
        # Find threshold score to keep top-k
        threshold = torch.kthvalue(scores, k).values
        mask = scores >= threshold

        return mask
