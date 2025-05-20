print("Flag 0")
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


torch.seed.random()

seed = 42
torch.manual_seed(seed)
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("Flag 1")

transform = transforms.Compose([
    transforms.RandomAffine(degrees=35, translate=(0.1, 0.1)),  # small rotation and shift
    transforms.ToTensor(),
])

dataset1 = datasets.MNIST('data', train=True, download=True, transform=transform)
dataset2 = datasets.MNIST('data', train=False, transform=transform)

print("Flag 2")

BATCH_SIZE = 32
EPOCHS = 12
NUM_MASKS = 1
LR = 0.001

seed = 42
torch.manual_seed(seed)
train_dataloader = DataLoader(dataset1, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
test_dataloader = DataLoader(dataset2, batch_size=BATCH_SIZE, pin_memory=True)

model = NetNormalDropoutV2().to(device)
opt = Adam(model.parameters(), lr=LR)
lossFn = torch.nn.NLLLoss()  # Use NLL since our model is outputting a probability

for i in range(EPOCHS):
    model.train()
    trainCorrect = 0
    totalLoss = 0
    print(f"Epoch {i}", flush=True)
    for idx, (x, y) in enumerate(train_dataloader):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = lossFn(logits, y)
        totalLoss += loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
        trainCorrect += (logits.argmax(1) == y).float().sum().item()
    print(f"Train Accuracy: {trainCorrect / len(dataset1):.4f}")
    print(f"Total loss: {totalLoss:.4f}")

test_correct = 0
model.eval()
with torch.no_grad():
    for idx, (x, y) in tqdm(enumerate(test_dataloader)):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = torch.argmax(logits, dim=1)
        test_correct += (pred == y).sum().item()
print(f"Test Accuracy: {test_correct / len(dataset2):.4f}")


