
import sys
sys.path.append('../')
import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from model import NetV2, MCMC
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from torch.optim import Adam
import os
from NISP import NISP
import torch.nn.utils.prune as prune


transform = transforms.Compose([
    transforms.RandomAffine(degrees=35, translate=(0.1, 0.1)),  # small rotation and shift
    transforms.ToTensor(),
])

dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)



BATCH_SIZE = 32
EPOCHS = 10
NUM_MASKS = 10
LR = 0.001
dropout_probs=[0.4, 0.7]

seed = 42
torch.manual_seed(seed)
indices = torch.randperm(len(dataset1)).tolist()  # Shuffled once

# Use SubsetRandomSampler to keep the same shuffle order on each epoch for the over-fitting step
sampler = SequentialSampler(indices)  # Keeps the order fixed
train_dataloader = DataLoader(dataset1, batch_size=BATCH_SIZE, sampler=sampler)
test_dataloader = DataLoader(dataset2, batch_size=1)

model_iteration = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not os.path.exists(f"../model{model_iteration}.pth"):
    model = NetV2(num_masks=NUM_MASKS, dropout_probs=dropout_probs).to(device)
    opt = Adam(model.parameters(), lr=LR)
    lossFn = torch.nn.NLLLoss() # Use NLL since we our model is outputting a probability
    torch.save(model.state_dict(), f"../model{model_iteration}.pth")
model_iteration += 1


train_accuracies_before_prune = []
train_losses_before_prune = []

if not os.path.exists(f"../model{model_iteration}.pth"):
    NGROUPS = 10 # dividing groups to use
    mask_groups = [list(range(i, NUM_MASKS, NGROUPS)) for i in range(NGROUPS)]  # Partition masks

    for epoch in range(EPOCHS):
        print(f"EPOCH: {epoch}")
        model.train()
        trainCorrect = 0
        totalLoss = 0
        tot = 0
        for idx, (x, y) in (enumerate(train_dataloader)):
            x, y = x.to(device), y.to(device)
            group_id = idx % NGROUPS  # Assign batch to a group
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
        acc = trainCorrect / tot
        print(f"Train Accuracy: {trainCorrect} / {tot}: {acc}")
        print(f"Total loss: {totalLoss}")
        train_accuracies_before_prune.append(acc)
        train_losses_before_prune.append(totalLoss)
    torch.save(model.state_dict(), f"../model{model_iteration}.pth")


model_iteration += 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_correct = 0
model.eval()
for idx, (x, y)  in (enumerate(test_dataloader)):
    # logits2 = model.forward(x, mask=1)
    x, y = x.to(device), y.to(device)
    logits = model.forward(x, mask=0)
    pred = torch.argmax(logits, dim=1)
    test_correct += (pred == y).sum().item()
print(f"Test accuracy on initial model: {test_correct / len(dataset2)}")

model = NetV2(num_masks=NUM_MASKS, dropout_probs=dropout_probs).to(device)
model.load_state_dict(torch.load(f"../model{1}.pth", map_location=device), strict=True)

# mcmc = MCMC(model=model, increment_amt=10)

seed = 42
torch.manual_seed(seed)
train_dataloader = DataLoader(dataset1, batch_size=BATCH_SIZE, shuffle=True)

# model.eval()
# for i in range(1):
#     trainCorrect = 0
#     for idx, (x, y)  in tqdm(enumerate(train_dataloader)):
#         mcmc.transition(x=x, y=y)


# # %%
# dist = torch.tensor([(val / mcmc.tot).item() for val in mcmc.ocurrences])
# print(dist)

# values, indices = torch.topk(dist, k=10)
# print(indices)
# print(indices.shape)

# test_correct_top_3 = 0
# test_correct = 0
# model.eval()
# for idx, (x, y)  in tqdm(enumerate(test_dataloader)):
#     # logits2 = model.forward(x, mask=1)
#     logits = mcmc.predict(x, chosen_masks=indices)
#     pred = torch.argmax(logits, dim=1)
#     test_correct += (pred == y).sum().item()


#     logits = mcmc.predict(x, chosen_masks=indices)
#     pred = torch.argmax(logits, dim=1)
#     test_correct_top_3 += (pred == y).sum().item()
# print(f"Test accuracy with all masks: {test_correct / len(dataset2)}")
# print(f"Test accuracy with top 3: {test_correct_top_3 / len(dataset2)}")

# # Pruning
# 

reshape_layers = {
    'pool': (64, 24, 24),
    'flat': (64, 12, 12)
}
    
dropout_layers = {
    'conv2': model.dropout1,
    'fc1': model.dropout2,
}

nisp = NISP(model, dropout_dict=dropout_layers, resize_dict=reshape_layers)
importance_scores = nisp.compute_aggregated_importance_scores()

print(model.conv1.weight.count_nonzero(), model.conv1.bias.count_nonzero())
print(model.conv2.weight.count_nonzero(), model.conv2.bias.count_nonzero())
print(model.fc1.weight.count_nonzero(), model.fc1.bias.count_nonzero())

initial_model = NetV2(num_masks=1, dropout_probs=[0.0, 0.0]).to(device)
initial_model.load_state_dict(torch.load(f"../model{0}.pth", map_location=device))
conv1_scores = importance_scores['conv1']
conv1_mask = nisp.get_pruning_mask(conv1_scores, pruning_rate=0.2)

conv2_scores = importance_scores['conv2']
conv2_mask = nisp.get_pruning_mask(conv2_scores, pruning_rate=0.4)


# Let's say you've already computed the scores and pruning mask:
fc1_scores = importance_scores['fc1']
fc1_mask = nisp.get_pruning_mask(fc1_scores, pruning_rate=0.7)

# %%
nisp.apply_nisp_pruning(initial_model, 'conv1', conv1_mask)
nisp.apply_nisp_pruning(initial_model, 'conv2', conv2_mask)
nisp.apply_nisp_pruning(initial_model, 'fc1', fc1_mask)

# %%
prune.remove(initial_model.conv1, 'weight')
# prune.remove(initial_model.conv1, 'bias')
prune.remove(initial_model.conv2, 'weight')
# prune.remove(initial_model.conv2, 'bias')
prune.remove(initial_model.fc1, 'weight')
# prune.remove(initial_model.fc1, 'bias')

print(initial_model.conv1.weight.count_nonzero(), initial_model.conv1.bias.count_nonzero())
print(initial_model.conv2.weight.count_nonzero(), initial_model.conv2.bias.count_nonzero())
print(initial_model.fc1.weight.count_nonzero(), initial_model.fc1.bias.count_nonzero())

model = initial_model
opt = Adam(model.parameters(), lr=LR)
lossFn = torch.nn.NLLLoss() # Use NLL since we our model is outputting a probability

model_iteration = 2

model  = initial_model
train_accuracies_after_prune = []
train_losses_after_prune = []
if not os.path.exists(f"../model{model_iteration}.pth"):
    NGROUPS = 1 # dividing groups to use
    mask_groups = [list(range(i, NUM_MASKS, NGROUPS)) for i in range(NGROUPS)]  # Partition masks

    for epoch in range(EPOCHS):
        model.train()
        trainCorrect = 0
        totalLoss = 0
        tot = 0
        for idx, (x, y) in (enumerate(train_dataloader)):
            x, y = x.to(device), y.to(device)
            group_id = idx % NGROUPS  # Assign batch to a group
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
        acc = trainCorrect / tot
        print(f"Train Accuracy: {trainCorrect} / {tot}: {trainCorrect / tot}")
        print(f"Total loss: {totalLoss}")
        train_accuracies_before_prune.append(acc)
        train_losses_before_prune.append(totalLoss)
    torch.save(model.state_dict(), f"../model{model_iteration}.pth")
    model1 = model

# Plotting Loss and Accuracy
plt.figure(figsize=(12, 5))

# Loss
plt.figure()
plt.plot(train_losses_before_prune, label="Train loss before pruning", color="red")
plt.plot(train_losses_after_prune, label="Train loss before pruning", color="blue")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss per Epoch")
plt.legend()
plt.savefig('training_loss.png')
plt.close()

# Accuracy
plt.figure()
plt.plot(train_accuracies_before_prune, label="Train Accuracy before pruning", color="red")
plt.plot(train_accuracies_after_prune, label="Train Accuracy after pruning", color="blue")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy per Epoch")
plt.legend()
plt.savefig("training_accuracy.png")
plt.close()

test_correct = 0
model.eval()
for idx, (x, y)  in (enumerate(test_dataloader)):
    # logits2 = model.forward(x, mask=1)
    x, y = x.to(device), y.to(device)
    logits = model.forward(x, mask=0)
    pred = torch.argmax(logits, dim=1)
    test_correct += (pred == y).sum().item()
print(f"Test accuracy with pruning: {test_correct / len(dataset2)}")
# print(f"Test accuracy with top 3: {test_correct_top_3 / len(dataset2)}")