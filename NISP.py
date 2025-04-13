import torch
import torch.nn as nn
from typing import Dict
from model import ConsistentMCDropout  # Assuming this is a custom module you have


class NISP:
    def __init__(self, model: nn.Module, dropout_dict, resize_dict):
        self.model = model
        self.importance_scores = {}
        self.dropout_dict = dropout_dict
        self.resize_dict = resize_dict

    def compute_aggregated_importance_scores(self) -> Dict[str, torch.Tensor]:
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
        mask_keys = list(next(iter(self.dropout_dict.values())).mask_dict.keys())

        for mask_id in mask_keys:
            scores = self.compute_importance_scores_for_mask(
                mask_id=mask_id,
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

    def compute_importance_scores_for_mask(self, mask_id: int) -> Dict[str, torch.Tensor]:
        """
        Compute importance scores for a single dropout mask.

        Args:

            mask_id: which mask index to use
            final_layer_scores: optional tensor for final layer importance

        Returns:
            Dict[layer_name → importance scores]
        """
        # Collect and reverse layer order
        layers = [(name, module) for name, module in self.model.named_modules()
                  if isinstance(module, (nn.Linear, nn.Conv2d, nn.MaxPool2d, nn.Flatten))]
        
        layers.reverse()

        importance_scores = {}

        # Initialize with uniform score in final layer

        final_layer = layers[0][1]
        if isinstance(final_layer, nn.Linear):
            importance_scores[layers[0][0]] = torch.ones(final_layer.out_features)
        elif isinstance(final_layer, nn.Conv2d):
            importance_scores[layers[0][0]] = torch.ones(final_layer.out_channels)


        transforms = []
        print(layers)
        for i in range(1, len(layers)):

            curr_name, curr_module = layers[i]
            prev_name, prev_module = layers[i - 1]

            print(curr_name, prev_name)
            print(curr_module, prev_module)
                        # Get dropout mask (or all-ones if not found)
            if prev_name in self.dropout_dict:
                mask = self.dropout_dict[prev_name].mask_dict[mask_id].float()
            else:
                mask = torch.ones_like(importance_scores[prev_name])
            
            if isinstance(prev_module, nn.Flatten):
                # Handle flatten layer
                if prev_name in self.resize_dict:
                    prev_scores = importance_scores[prev_name]
                    curr_scores = (prev_scores * mask).view(self.resize_dict[prev_name])
                    importance_scores[curr_name] = curr_scores
                continue
            elif isinstance(prev_module, nn.MaxPool2d):
                if prev_name in self.resize_dict:
                    prev_scores = importance_scores[prev_name]
                    curr_scores = torch.nn.functional.max_unpool2d(
                        prev_scores * mask, self.model.pool_indices['pool'].squeeze(dim=0), prev_module.kernel_size,
                        prev_module.stride, prev_module.padding
                    )
                    importance_scores[curr_name] = curr_scores
                continue


            weights = prev_module.weight
            if isinstance(prev_module, nn.Conv2d):
                weights = prev_module.weight  # [64, 32, 3, 3]
                out_channels, in_channels, kH, kW = weights.shape
                weights_flat = weights.view(out_channels, -1)

                # Get the output importance scores (e.g., from next layer)
                prev_scores = importance_scores[prev_name]
                masked_scores = prev_scores * mask

                 # Reduce spatial map to get per-output-channel score: [64
                reduced_scores = masked_scores.sum(dim=(1, 2))
                print("reducing", masked_scores.shape, reduced_scores.shape)

                input_scores_flat = torch.matmul(torch.abs(weights_flat.T), reduced_scores)  # [288]

                # Reshape to [in_channels, kH, kW]
                input_scores = input_scores_flat.view(in_channels, kH, kW)

                # Optionally reduce to per-channel importance
                input_channel_scores = input_scores.sum(dim=(1, 2))  # [in_channels]

                importance_scores[curr_name] = input_channel_scores
                continue

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
