import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from prepareDataset import test_loader 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)

model.load_state_dict(torch.load("resnet18_finetuned_V2.pth", map_location=device))
model = model.to(device)

model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

try:
    class_names = test_loader.dataset.dataset.classes
except AttributeError:
    class_names = [f"Klasa {i}" for i in range(10)]

print("\n--- FINAL REPORT ---")
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print(report)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.ylabel('Rzeczywista Etykieta Terenu')
plt.xlabel('Predykcja Sieci')
plt.title('Macierz Pomyłek - Zbiór Testowy')
plt.tight_layout()

plt.savefig('confusion_matrix.pdf', dpi=300)
print("done")