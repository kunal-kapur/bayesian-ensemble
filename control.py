# %%
import sys
sys.path.append('../')
import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from models_control import NetNormalDropoutV2
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam

# %% [markdown]
# ### Control experiment: Not using masks and no dropout
# This notebook evaluates the performance oof using a standard deep-net with 'normal' dropout layers in training which are removed at inferene

# %%
transform = transforms.Compose([
    transforms.RandomRotation(degrees=80),
    transforms.ToTensor(), transforms.GaussianBlur(kernel_size=5, sigma=(4, 5)),  # Stronger blur
])

# %%
dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)


# %%
BATCH_SIZE = 32
EPOCHS = 12
NUM_MASKS = 1
LR = 0.001

# %%
seed = 42
torch.manual_seed(seed)
train_dataloader = DataLoader(dataset1, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset2)

# %%
model = NetNormalDropoutV2()
opt = Adam(model.parameters(), lr=LR)
lossFn = torch.nn.NLLLoss() # Use NLL since we our model is outputting a probability


# %%
for i in range(EPOCHS):
    model.train()
    trainCorrect = 0
    totalLoss = 0
    for idx, (x, y)  in tqdm(enumerate(train_dataloader)):
        logits = model.forward(x)
        loss = lossFn(logits, y)
        totalLoss += loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
        trainCorrect += (logits.argmax(1) == y).type(
			torch.float).sum().item()
    print(f"Train Accuracy: {trainCorrect/len(dataset1)}")
    print(f"Total loss: {totalLoss}")

# %%
test_correct = 0
model.eval()
for idx, (x, y)  in tqdm(enumerate(test_dataloader)):
    logits = model.forward(x)
    pred = torch.argmax(logits, dim=1)
    test_correct += (pred == y).sum().item()
print(test_correct / len(dataset2))

# %%


# %%



