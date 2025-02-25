import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from model import Net, MCMC
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from torch.optim import Adam

transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

dataset1 = datasets.MNIST('data', train=True, download=True,
                       transform=transform)
dataset2 = datasets.MNIST('data', train=False,
                       transform=transform)

BATCH_SIZE = 32
EPOCHS = 10
NUM_MASKS = 10
LR = 0.001

seed = 42
torch.manual_seed(seed)
indices = torch.randperm(len(dataset1)).tolist()  # Shuffled once

# Use SubsetRandomSampler to keep the same shuffle order on each epoch for the over-fitting step
sampler = SequentialSampler(indices)  # Keeps the order fixed
train_dataloader = DataLoader(dataset1, batch_size=BATCH_SIZE, sampler=sampler)
test_dataloader = DataLoader(dataset2)

model = Net(num_masks=NUM_MASKS)
mcmc = MCMC(model=model, num_masks=NUM_MASKS, increment_amt=1)
opt = Adam(model.parameters(), lr=LR)
lossFn = torch.nn.NLLLoss() # Use NLL since we our model is outputting a probability

for i in range(EPOCHS):
    model.train()
    trainCorrect = 0
    for idx, (x, y)  in enumerate(train_dataloader):
        pred = model.forward(x, mask = idx % NUM_MASKS)
        loss = lossFn(torch.log(pred), y)
        opt.zero_grad()
        loss.backward()
        opt.step()


        trainCorrect += (pred.argmax(1) == y).type(
			torch.float).sum().item()
    print(f"Train Accuracy: {trainCorrect/len(dataset1)}")

