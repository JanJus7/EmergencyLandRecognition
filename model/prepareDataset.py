import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader, Dataset

torch.manual_seed(42)

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=90),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("EuroSat downloading...")
full_dataset = datasets.EuroSAT(root="./data", download=True, transform=None)
print(f"Done, {len(full_dataset)} photos loaded into 10 classes.")

dataset_size = len(full_dataset)

train_size = int(0.70 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size

train_subset, val_subset, test_subset = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42),
)

class DatasetWithTransform(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

train_dataset = DatasetWithTransform(train_subset, transform=train_transforms)
val_dataset = DatasetWithTransform(val_subset, transform=val_test_transforms)
test_dataset = DatasetWithTransform(test_subset, transform=val_test_transforms)

train_dataset.classes = full_dataset.classes
val_dataset.classes = full_dataset.classes
test_dataset.classes = full_dataset.classes

print("\nData split:")
print(f"Train:      {len(train_dataset)} photos")
print(f"Validation: {len(val_dataset)} photos")
print(f"Test:       {len(test_dataset)} photos")

batch_size = 32

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
)

val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
)

test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
)

print("\nDataLoaders created.")

images, labels = next(iter(train_loader))
print(f"\nbatch size (images): {images.shape}")
print(f"batch size (labels): {labels.shape}")

class_names = full_dataset.classes

print("\nclasses:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")
