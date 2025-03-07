import sys
import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from model import NetV2, MCMC
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from torch.optim import Adam
import os
import argparse
import math
import pandas as pd

transform = transforms.Compose([
    transforms.RandomRotation(degrees=180),
    transforms.ToTensor(), transforms.GaussianBlur(kernel_size=7, sigma=(4, 5)),  # Stronger blur
    transforms.Lambda(lambda x: torch.flatten(x)),
])

dataset1 = datasets.MNIST('data', train=True, download=True,
                       transform=transform)
dataset2 = datasets.MNIST('data', train=False,
                       transform=transform)

parser = argparse.ArgumentParser()

parser.add_argument('-e', "--epochs", type=int, default=12, help="Number of training epochs")    
parser.add_argument('-b', "--batch_size", type=int, default=32, help="Batch size for training")    
parser.add_argument('-l', "--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument('-nm', "--num_masks", type=int, required=True, help="Number of masks")   
parser.add_argument('-dp', "--dropout_probs", type=float, nargs='+', required=True, help="List of dropout probabilities")
parser.add_argument('-ng', "--num_groups", type=int, required=True, help="Number of groups for masks to divide amongst")
parser.add_argument('-i', "--increment_amt", type=int, default=1, help="amount to increment by")
parser.add_argument('-p', "--path", type=str, default="experiments", help="path to place experiment in")

# Parse arguments and read into variables
args = parser.parse_args()

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LR = args.lr
num_masks = args.num_masks
dropout_probs = args.dropout_probs
num_groups = args.num_groups
increment_amt = args.increment_amt
PATH = os.path.join(args.path, f"numMasks{num_masks}_numGroups{num_groups}_dropout_probs{dropout_probs}_increment_amt{increment_amt}")

if not os.path.exists(PATH):
    os.makedirs(PATH, exist_ok=True)



print(f"Epochs: {EPOCHS}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Learning Rate: {LR}")
print(f"Num Masks: {num_masks}")
print(f"Dropout Probs: {dropout_probs}")
print(f"Num Groups: {num_groups}")

SEED = 42

torch.manual_seed(SEED)


def unpack_arguments(**kwargs):
    return kwargs.get("batch_size", 32), kwargs.get("epochs", 12), kwargs.get("lr", 0.001)



indices = torch.randperm(len(dataset1)).tolist()  # Shuffled once
# Use SubsetRandomSampler to keep the same shuffle order on each epoch for the over-fitting step
sampler = SequentialSampler(indices)  # Keeps the order fixed
train_dataloader = DataLoader(dataset1, batch_size=BATCH_SIZE, sampler=sampler)
test_dataloader = DataLoader(dataset2, batch_size=1)


model = NetV2(num_masks=num_masks, dropout_probs=dropout_probs)
opt = Adam(model.parameters(), lr=LR)
lossFn = torch.nn.NLLLoss() # Use NLL since we our model is outputting a probability
mask_groups = [list(range(i, num_masks, num_groups)) for i in range(num_groups)]  # Partition masks

print("Beginning Training\n")
for epoch in tqdm(range(EPOCHS)):
    model.train()
    trainCorrect = 0
    totalLoss = 0
    tot = 0
    for idx, (x, y) in (enumerate(train_dataloader)):
        group_id = idx % num_groups  # Assign batch to a group
        masks = mask_groups[group_id]  # Get all masks in this group
        
        for mask in masks:
            logits = model.forward(x, mask=mask)
            loss = lossFn(logits, y)
            totalLoss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
            trainCorrect += (logits.argmax(1) == y).type(torch.float).sum().item()
            tot += len(y)


print("\n\nEstimating discrete distribution using MCMC\n")
mcmc = MCMC(model=model, increment_amt=increment_amt)
train_dataloader = DataLoader(dataset1, batch_size=BATCH_SIZE, shuffle=True)
model.eval()
for i in range(1):
    trainCorrect = 0
    for idx, (x, y)  in tqdm(enumerate(train_dataloader)):
        mcmc.transition(x=x, y=y)

dist = mcmc.recalculate_dist()
values, indices = torch.topk(dist, k=num_masks)

porportions = [0.2, 0.4, 0.8, 1]

# taking a portion of the top masks instead of incrementing by 1
tot_masks_used = [1]
[tot_masks_used.append(math.floor(porp * num_masks)) for porp in porportions]
tot_masks_used = sorted(list(set(tot_masks_used)))


print("\n\n")


print("Testing on different number of masks\n")
test_acc = []
for i in tqdm(tot_masks_used):
    indices_used = indices[0:i]
    test_correct = 0
    model.eval()
    for idx, (x, y)  in tqdm(enumerate(test_dataloader)):
        logits = mcmc.predict(x, chosen_masks=indices_used)
        pred = torch.argmax(logits, dim=1)
        test_correct += (pred == y).sum().item()
    test_acc.append(test_correct / len(dataset2))

# Plot Test Accuracy vs. Num Masks
plt.plot(tot_masks_used, test_acc, label="Accuracy")
plt.xlabel("Top k Masks used")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy vs. Number of Masks")
plt.legend()
plt.savefig(os.path.join(PATH, "acc_per_mask.png"))

# Save results to a CSV file
df = pd.DataFrame({
    "num_masks": tot_masks_used,
    "test_accuracy": test_acc,
    "EPOCHS": [EPOCHS] * len(tot_masks_used),
    "BATCH_SIZE": [BATCH_SIZE] * len(tot_masks_used),
    "LR": [LR] * len(tot_masks_used),
    "dropout_probs1": [dropout_probs[0]] * len(tot_masks_used),
    "dropout_probs2": [dropout_probs[1]] * len(tot_masks_used),
    "num_groups": [num_groups] * len(tot_masks_used),
    "increment_amt": [increment_amt] * len(tot_masks_used),
})

csv_path = os.path.join(PATH, "results.csv")
df.to_csv(csv_path, index=False)

print(f"Saved results to {csv_path}")

