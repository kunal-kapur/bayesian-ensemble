import torch
import torch.nn as nn
from typing import Dict
from model import ConsistentMCDropout  # Assuming this is a custom module you have
from torch.nn.utils import prune


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
        layers = [(name, module) for name, module in self.model.named_modules()
                if isinstance(module, (nn.Linear, nn.Conv2d, nn.Flatten))]
        layers.reverse()

        device = next(self.model.parameters()).device
        importance_scores = {}

        final_layer = layers[0][1]
        if isinstance(final_layer, nn.Linear):
            importance_scores[layers[0][0]] = torch.ones(final_layer.out_features, device=device)
        elif isinstance(final_layer, nn.Conv2d):
            importance_scores[layers[0][0]] = torch.ones(final_layer.out_channels, device=device)

        for i in range(1, len(layers)):
            curr_name, curr_module = layers[i]
            prev_name, prev_module = layers[i - 1]

            # Dropout mask
            if prev_name in self.dropout_dict:
                mask = self.dropout_dict[prev_name].mask_dict[mask_id].squeeze(dim=0).float().to(device)
            else:
                mask = torch.ones_like(importance_scores[prev_name]).to(device)

            if isinstance(prev_module, nn.Flatten):
                if prev_name in self.resize_dict:
                    prev_scores = importance_scores[prev_name].to(device)
                    curr_scores = (prev_scores * mask).view(self.resize_dict[prev_name])
                    curr_scores = curr_scores.sum(dim=(1, 2))
                    curr_scores = curr_scores.view(-1, 1, 1)
                    importance_scores[curr_name] = curr_scores
                continue

            elif isinstance(prev_module, nn.MaxPool2d):
                if prev_name in self.resize_dict:
                    prev_scores = importance_scores[prev_name].to(device)
                    importance_scores[curr_name] = prev_scores
                continue

            weights = prev_module.weight.to(device)

            prev_scores = importance_scores[prev_name].to(device)

            if isinstance(prev_module, nn.Conv2d):
                out_channels, in_channels, kH, kW = weights.shape
                weights_flat = weights.sum(dim=(2, 3))  # shape: [out_ch, in_ch]
                masked_scores = (prev_scores * mask).squeeze()
                input_scores_flat = torch.matmul(torch.abs(weights_flat.T), masked_scores)
                importance_scores[curr_name] = input_scores_flat.view(in_channels, 1, 1)
                continue

            # Linear case
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
        scores = scores.squeeze()
        k = int(scores.numel() * (1 - pruning_rate))
        if k <= 0:
            return torch.zeros_like(scores, dtype=torch.bool)
        
        # Find threshold score to keep top-k
        threshold = torch.kthvalue(scores, k).values
        mask = scores >= threshold

        return mask
    
    def apply_nisp_pruning(self, model: nn.Module, layer_name: str, neuron_mask: torch.Tensor):
        """
        Applies structured pruning to a given layer using a neuron-wise mask.
        Supports both nn.Linear and nn.Conv2d layers.

        Args:
            model: The PyTorch model.
            layer_name: Name of the layer to prune (e.g., 'fc1' or 'conv1').
            neuron_mask: 1D tensor with 1s for neurons/channels to keep, 0s to prune.
        """
        layer = dict(model.named_modules())[layer_name]
        
        if isinstance(layer, nn.Conv2d):
            weight_mask = neuron_mask[:, None, None, None].expand_as(layer.weight.data)
        elif isinstance(layer, nn.Linear):
            weight_mask = neuron_mask[:, None].expand_as(layer.weight.data)
        else:
            raise TypeError(f"Unsupported layer type: {type(layer)}")

        prune.CustomFromMask.apply(layer, name='weight', mask=weight_mask)
        if layer.bias is not None:
            bias_mask = neuron_mask.clone()
            prune.CustomFromMask.apply(layer, name='bias', mask=bias_mask)


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
        layers = [(name, module) for name, module in self.model.named_modules()
                if isinstance(module, (nn.Linear, nn.Conv2d, nn.Flatten))]
        layers.reverse()

        device = next(self.model.parameters()).device
        importance_scores = {}

        final_layer = layers[0][1]
        if isinstance(final_layer, nn.Linear):
            importance_scores[layers[0][0]] = torch.ones(final_layer.out_features, device=device)
        elif isinstance(final_layer, nn.Conv2d):
            importance_scores[layers[0][0]] = torch.ones(final_layer.out_channels, device=device)

        for i in range(1, len(layers)):
            curr_name, curr_module = layers[i]
            prev_name, prev_module = layers[i - 1]

            # Dropout mask
            if prev_name in self.dropout_dict:
                mask = self.dropout_dict[prev_name].mask_dict[mask_id].squeeze(dim=0).float().to(device)
            else:
                mask = torch.ones_like(importance_scores[prev_name]).to(device)

            if isinstance(prev_module, nn.Flatten):
                if prev_name in self.resize_dict:
                    prev_scores = importance_scores[prev_name].to(device)
                    curr_scores = (prev_scores * mask).view(self.resize_dict[prev_name])
                    curr_scores = curr_scores.sum(dim=(1, 2))
                    curr_scores = curr_scores.view(-1, 1, 1)
                    importance_scores[curr_name] = curr_scores
                continue

            elif isinstance(prev_module, nn.MaxPool2d):
                if prev_name in self.resize_dict:
                    prev_scores = importance_scores[prev_name].to(device)
                    importance_scores[curr_name] = prev_scores
                continue

            weights = prev_module.weight.to(device)

            prev_scores = importance_scores[prev_name].to(device)

            if isinstance(prev_module, nn.Conv2d):
                out_channels, in_channels, kH, kW = weights.shape
                weights_flat = weights.sum(dim=(2, 3))  # shape: [out_ch, in_ch]
                masked_scores = (prev_scores * mask).squeeze()
                input_scores_flat = torch.matmul(torch.abs(weights_flat.T), masked_scores)
                importance_scores[curr_name] = input_scores_flat.view(in_channels, 1, 1)
                continue

            # Linear case
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
        
        scores = scores.squeeze()
        k = int(scores.numel() * (1 - pruning_rate))
        if k <= 0:
            return torch.zeros_like(scores, dtype=torch.bool)

        # Get indices of the top-k scores (highest scores to keep)
        topk_indices = torch.topk(scores, k).indices
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask[topk_indices] = True
        return mask
    
    def apply_nisp_pruning(self, model: nn.Module, layer_name: str, neuron_mask: torch.Tensor):
        """
        Applies structured pruning to a given layer using a neuron-wise mask.
        Supports both nn.Linear and nn.Conv2d layers.

        Args:
            model: The PyTorch model.
            layer_name: Name of the layer to prune (e.g., 'fc1' or 'conv1').
            neuron_mask: 1D tensor with 1s for neurons/channels to keep, 0s to prune.
        """
        layer = dict(model.named_modules())[layer_name]
        
        if isinstance(layer, nn.Conv2d):
            weight_mask = neuron_mask[:, None, None, None].expand_as(layer.weight.data)
        elif isinstance(layer, nn.Linear):
            weight_mask = neuron_mask[:, None].expand_as(layer.weight.data)
        else:
            raise TypeError(f"Unsupported layer type: {type(layer)}")

        prune.CustomFromMask.apply(layer, name='weight', mask=weight_mask)
        if layer.bias is not None:
            bias_mask = neuron_mask.clone()
            prune.CustomFromMask.apply(layer, name='bias', mask=bias_mask)

