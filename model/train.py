import os
import torch
from torch import nn
from torchvision import models
from torch.utils.data import DataLoader
from prepareDataset import train_loader, val_loader, test_loader
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

patience = 5
delta = 0.001


class EarlyStopping:
    def __init__(self, patience=5, delta=0.01, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = None
        self.no_improvement_count = 0
        self.early_stop = False

    def check_early_stop(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.no_improvement_count = 0
            torch.save(model.state_dict(), "resnet18_finetuned.pth")
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0
            torch.save(model.state_dict(), "resnet18_finetuned.pth")
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Stopping early as no improvement has been observed.")


model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False

for param in model.layer3.parameters():
    param.requires_grad = True
for param in model.layer4.parameters():
    param.requires_grad = True

model.fc = nn.Linear(model.fc.in_features, 10)

model = model.to(device)

print("Layer Status (True = Learning, False = Frozen):")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: requires_grad={param.requires_grad}")

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

early_stopping = EarlyStopping(patience=patience, delta=delta, verbose=True)

with open("training_history.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Train_Loss", "Val_Loss", "Val_Accuracy"])

for epoch in range(1, 100):

    train_loss = 0.0
    val_loss = 0.0

    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        
    epoch_loss = running_loss / len(train_loader.dataset)

    print(f"Epoch {epoch}/100, Loss: {epoch_loss:.4f}")

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            val_loss += loss.item() * images.size(0)

    accuracy = correct / total
    val_loss /= len(val_loader.dataset)

    early_stopping.check_early_stop(val_loss, model)

    with open("training_history.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, epoch_loss, val_loss, accuracy])

    print(f"Epoch {epoch}/100, Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.4f}")

    if early_stopping.early_stop:
        print(f"Early stopping at epoch {epoch}")
        break
