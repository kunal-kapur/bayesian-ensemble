
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
import os
import time
import csv
import argparse

timestamp = time.strftime("%Y-%m-%d--%H-%M-%S")
print("Timestamp", timestamp)
EXPERIMENT_FOLDER = f"experiments/{timestamp}"
if not os.path.exists(EXPERIMENT_FOLDER):
    os.makedirs(EXPERIMENT_FOLDER)



for i in range(0, 3):
    if os.path.exists(f"model{i}.pth"):
        os.remove(f"model{i}.pth")



transform = transforms.Compose([
    transforms.RandomAffine(degrees=35, translate=(0.1, 0.1)),  # small rotation and shift
    transforms.ToTensor(),
])

dataset1 = datasets.MNIST('data', train=True, download=True,
                       transform=transform)
dataset2 = datasets.MNIST('data', train=False,
                       transform=transform)

parser = argparse.ArgumentParser()
parser.add_argument('-e', "--epochs", type=int, default=12, help="Number of training epochs")    
parser.add_argument('-b', "--batch_size", type=int, default=32, help="Batch size for training")    
parser.add_argument('-l', "--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument('-nm', "--num_masks", type=int, default=10,  required=True, help="Number of masks")   
parser.add_argument('-dp', "--dropout_probs", type=float, nargs='+', required=True, help="List of dropout probabilities")
parser.add_argument('-ng', "--num_groups", type=int, required=True, help="Number of groups for masks to divide amongst")
parser.add_argument('-p', "--pruning_rate", type=float, nargs='+', required=True, help="amount to prune the given layer")

# Parse arguments and read into variables
args = parser.parse_args()

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LR = args.lr
NUM_MASKS = args.num_masks
dropout_probs = args.dropout_probs
NGROUPS = args.num_groups
PRUNING_RATE = args.pruning_rate

with open(os.path.join(EXPERIMENT_FOLDER, "hyerpparams.txt"), 'w') as f:
    f.write(f"Batch size: {BATCH_SIZE}\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write(f"Learning rate: {LR}\n")
    f.write(f"Dropout probabilties: {dropout_probs}\n")
    f.write(f"Number of groups: {NGROUPS}\n")
    f.write(f"Pruning rates: {PRUNING_RATE}\n")

seed = 42
torch.manual_seed(seed)
indices = torch.randperm(len(dataset1)).tolist()  # Shuffled once

# Use SubsetRandomSampler to keep the same shuffle order on each epoch for the over-fitting step
sampler = SequentialSampler(indices)  # Keeps the order fixed
train_dataloader = DataLoader(dataset1, batch_size=BATCH_SIZE, sampler=sampler)
test_dataloader = DataLoader(dataset2, batch_size=1)

model_iteration = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not os.path.exists(os.path.join(EXPERIMENT_FOLDER, f"model{model_iteration}.pth")):
    model = NetV2(num_masks=NUM_MASKS, dropout_probs=dropout_probs).to(device)
    opt = Adam(model.parameters(), lr=LR)
    lossFn = torch.nn.NLLLoss() # Use NLL since we our model is outputting a probability
    torch.save(model.state_dict(), os.path.join(EXPERIMENT_FOLDER, f"model{model_iteration}.pth"))
model_iteration += 1


train_accuracies_before_prune = []
train_losses_before_prune = []

if not os.path.exists(os.path.join(EXPERIMENT_FOLDER, f"model{model_iteration}.pth")):
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
    torch.save(model.state_dict(), os.path.join(EXPERIMENT_FOLDER, f"model{model_iteration}.pth"))

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

print(f"Test accuracy on initial model: {test_correct / len(dataset2)}", flush=True)
test_acc_pre_prune = test_correct / len(dataset2)

model = NetV2(num_masks=NUM_MASKS, dropout_probs=dropout_probs).to(device)
model.load_state_dict(torch.load(os.path.join(EXPERIMENT_FOLDER, f"model{1}.pth"), map_location=device), strict=True)

# mcmc = MCMC(model=model, increment_amt=10)

seed = 42
torch.manual_seed(seed)
train_dataloader = DataLoader(dataset1, batch_size=BATCH_SIZE, shuffle=True)

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


print("Before pruning weights")
print(model.conv1.weight.count_nonzero(), model.conv1.bias.count_nonzero())
print(model.conv2.weight.count_nonzero(), model.conv2.bias.count_nonzero())
print(model.fc1.weight.count_nonzero(), model.fc1.bias.count_nonzero())

with open (os.path.join(EXPERIMENT_FOLDER, "pruning.txt"), 'w') as f:
    f.write(f"conv1: {model.conv1.weight.count_nonzero()}\n")
    f.write(f"conv2: {model.conv2.weight.count_nonzero()}\n")
    f.write(f"fc1: {model.fc1.weight.count_nonzero()}\n")
    f.write("-----------------------------------")
    f.write("\n\n")


initial_model = NetV2(num_masks=1, dropout_probs=[0.0, 0.0]).to(device)
initial_model.load_state_dict(torch.load(os.path.join(EXPERIMENT_FOLDER, f"model{0}.pth"), map_location=device))
conv1_scores = importance_scores['conv1']
conv1_mask = nisp.get_pruning_mask(conv1_scores, pruning_rate=PRUNING_RATE[0])

conv2_scores = importance_scores['conv2']
conv2_mask = nisp.get_pruning_mask(conv2_scores, pruning_rate=PRUNING_RATE[1])

fc1_scores = importance_scores['fc1']
fc1_mask = nisp.get_pruning_mask(fc1_scores, pruning_rate=PRUNING_RATE[2])

# creating mask
nisp.apply_nisp_pruning(initial_model, 'conv1', conv1_mask)
nisp.apply_nisp_pruning(initial_model, 'conv2', conv2_mask)
nisp.apply_nisp_pruning(initial_model, 'fc1', fc1_mask)


# pruning off weights
prune.remove(initial_model.conv1, 'weight')
prune.remove(initial_model.conv2, 'weight')
prune.remove(initial_model.fc1, 'weight')


print("\n\nAfter pruning weights", flush=True)
print(initial_model.conv1.weight.count_nonzero(), initial_model.conv1.bias.count_nonzero())
print(initial_model.conv2.weight.count_nonzero(), initial_model.conv2.bias.count_nonzero())
print(initial_model.fc1.weight.count_nonzero(), initial_model.fc1.bias.count_nonzero())

with open (os.path.join(EXPERIMENT_FOLDER, "pruning.txt"), 'a') as f:
    f.write("After Pruning\n")
    f.write(f"conv1: {initial_model.conv1.weight.count_nonzero()}\n")
    f.write(f"conv2: {initial_model.conv2.weight.count_nonzero()}\n")
    f.write(f"fc1: {initial_model.fc1.weight.count_nonzero()}\n")

model = initial_model
opt = Adam(model.parameters(), lr=LR)
lossFn = torch.nn.NLLLoss() # Use NLL since we our model is outputting a probability

model_iteration = 2

model  = initial_model
train_accuracies_after_prune = []
train_losses_after_prune = []
if not os.path.exists(os.path.join(EXPERIMENT_FOLDER, f"model{model_iteration}.pth")):
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
        train_accuracies_after_prune.append(acc)
        train_losses_after_prune.append(totalLoss)
    torch.save(model.state_dict(), os.path.join(EXPERIMENT_FOLDER, f"model{model_iteration}.pth"))
    model1 = model

# Plotting Loss and Accuracy
plt.figure(figsize=(12, 5))

# Loss
plt.figure()
plt.plot(train_losses_before_prune, label="Train loss before pruning", color="red")
plt.plot(train_losses_after_prune, label="Train loss after pruning", color="blue")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss per Epoch")
plt.legend()
plt.savefig(os.path.join(EXPERIMENT_FOLDER, 'training_loss.png'))
plt.close()

# Accuracy
plt.figure()
plt.plot(train_accuracies_before_prune, label="Train Accuracy before pruning", color="red")
plt.plot(train_accuracies_after_prune, label="Train Accuracy after pruning", color="blue")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy per Epoch")
plt.legend()
plt.savefig(os.path.join(EXPERIMENT_FOLDER, "training_accuracy.png"))
plt.close()


headers = ("epochs", "accuracy-pre-pruning",
        "loss-pre-pruning",
        "accuracy-post-pruning",
        "loss-post-pruning")

rows = zip([i for i in range(0, EPOCHS)],
    train_accuracies_before_prune, train_accuracies_after_prune, train_losses_before_prune, train_accuracies_after_prune)
filename = os.path.join(EXPERIMENT_FOLDER, "out.csv")
with open(filename, "w") as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    for row in rows:
        writer.writerow(row)

test_correct = 0
model.eval()
for idx, (x, y)  in (enumerate(test_dataloader)):
    # logits2 = model.forward(x, mask=1)
    x, y = x.to(device), y.to(device)
    logits = model.forward(x, mask=0)
    pred = torch.argmax(logits, dim=1)
    test_correct += (pred == y).sum().item()
print(f"Test accuracy with pruning: {test_correct / len(dataset2)}", flush=True)
test_acc_post_prune = test_correct / len(dataset2)

with open(os.path.join(EXPERIMENT_FOLDER, "Test-acc.txt"), 'w') as f:
    f.write(f"Accuracy before pruning: {test_acc_pre_prune}\n")
    f.write(f"Accuracy after pruning: {test_acc_post_prune}")

exit()