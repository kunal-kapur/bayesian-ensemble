import torch
import torch.nn.utils.prune as prune
from resnet import ResNet18  # Replace 'model_file' with your filename

def main():
    # 1. Initialize the model
    dropout_probs = [0.8, 0.8]
    model = ResNet18(dropout_probs=dropout_probs)
    model.eval()

    # 2. Compute NISP scores
    with torch.no_grad():
        scores = model.compute_nisp_scores()  # [total output channels across all conv layers]

    print(f"Computed NISP scores of length: {len(scores)}")

    # 3. Apply structured pruning
    pruning_rate = 0.4  # Prune 40% of channels globally
    print(f"Pruning with rate: {pruning_rate}")
    model.prune_with_scores(scores, pruning_rate)

    # (Optional) 4. Count zeroed-out channels
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            if hasattr(module, 'weight_mask'):
                pruned_channels = (module.weight_mask.sum(dim=(1, 2, 3)) == 0).sum().item()
                print(f"{name}: Pruned {pruned_channels}/{module.out_channels} channels")

    # (Optional) Save model
    torch.save(model.state_dict(), "resnet18_pruned.pth")

if __name__ == "__main__":
    main()
