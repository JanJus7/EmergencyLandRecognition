import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

torch.manual_seed(42)

data_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

print("EuroSat downloading...")
full_dataset = datasets.EuroSAT(root="./data", download=True, transform=data_transforms)

print(f"Done, {len(full_dataset)} photos loaded into 10 classes.")

dataset_size = len(full_dataset)

train_size = int(0.70 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42),
)

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
