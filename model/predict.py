import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
import matplotlib.pyplot as plt
import argparse

class_names = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


def slidingWindow(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y : y + windowSize[1], x : x + windowSize[0]])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(
    torch.load("resnet18_finetuned_V2.pth", map_location=device, weights_only=True)
)
model.to(device)
model.eval()

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

parser = argparse.ArgumentParser(description="Predykcja stref lądowania awaryjnego.")
parser.add_argument(
    "-a",
    "--aircraft",
    choices=["c152", "b737"],
    required=True,
    help="Wybór statku powietrznego: c152 (Cessna 152) lub b737 (Boeing 737-800)",
)
parser.add_argument(
    "-i",
    "--image",
    type=str,
    default="bay",
    help="Ścieżka do obrazu satelitarnego do analizy (domyślnie: bay)",
)
args = parser.parse_args()

if args.aircraft == "c152":
    print("Cessna 152 (Light General Aviation Aircraft) profile selected.")
    landing_scores = {
        "AnnualCrop": 0.9,
        "Forest": 0.0,
        "HerbaceousVegetation": 0.8,
        "Highway": 0.9,
        "Industrial": 0.0,
        "Pasture": 1.0,
        "PermanentCrop": 0.9,
        "Residential": 0.0,
        "River": 0.2,
        "SeaLake": 0.2,
    }
else:
    print("Boeing 737-800 (Heavy Jet Passenger Aircraft) profile selected.")
    landing_scores = {
        "AnnualCrop": 0.1,
        "Forest": 0.0,
        "HerbaceousVegetation": 0.1,
        "Highway": 0.8,
        "Industrial": 0.0,
        "Pasture": 0.0,
        "PermanentCrop": 0.1,
        "Residential": 0.0,
        "River": 0.6,
        "SeaLake": 0.5,
    }

imgPath = f"data/gEarth/{args.image}_acc_r.jpg"
image = cv2.imread(imgPath)

if image is None:
    print(f"Error: Could not read image from {imgPath}...")
    exit()

orig_h, orig_w = image.shape[0], image.shape[1]
winW, winH = (224, 224)

step_size = 112
batch_size = 64

print(f"Entry resolution: {image.shape[1]}x{image.shape[0]}")

pad_bottom = (step_size - ((image.shape[0] - winH) % step_size)) % step_size
pad_right = (step_size - ((image.shape[1] - winW) % step_size)) % step_size

image = cv2.copyMakeBorder(
    image, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
)

print(f"Resolution after padding: {image.shape[1]}x{image.shape[0]}")

heatmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
counter = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

batch_tensors = []
batch_coords = []

print("Starting terrain scanning...")

for x, y, window in slidingWindow(image, stepSize=step_size, windowSize=(winW, winH)):
    if window.shape[0] != winH or window.shape[1] != winW:
        continue

    window_rgb = cv2.cvtColor(window, cv2.COLOR_BGR2RGB)
    tensor = transform(window_rgb)

    batch_tensors.append(tensor.unsqueeze(0))
    batch_coords.append((x, y))

    if len(batch_tensors) >= batch_size:
        batch_data = torch.cat(batch_tensors, dim=0).to(device)

        with torch.no_grad():
            outputs = model(batch_data)

        probs = torch.softmax(outputs, dim=1)
        predicted_indices = torch.argmax(probs, dim=1)

        for i in range(len(batch_coords)):
            bx, by = batch_coords[i]
            predicted_class = class_names[predicted_indices[i].item()]
            score = landing_scores[predicted_class]

            heatmap[by : by + winH, bx : bx + winW] += score
            counter[by : by + winH, bx : bx + winW] += 1

        batch_tensors = []
        batch_coords = []

if len(batch_tensors) > 0:
    batch_data = torch.cat(batch_tensors, dim=0).to(device)
    with torch.no_grad():
        outputs = model(batch_data)

    probs = torch.softmax(outputs, dim=1)
    predicted_indices = torch.argmax(probs, dim=1)

    for i in range(len(batch_coords)):
        bx, by = batch_coords[i]
        predicted_class = class_names[predicted_indices[i].item()]
        score = landing_scores[predicted_class]

        heatmap[by : by + winH, bx : bx + winW] += score
        counter[by : by + winH, bx : bx + winW] += 1

print("Terrain scanning completed. Generating heatmap...")

heatmap = heatmap[:orig_h, :orig_w]
counter = counter[:orig_h, :orig_w]
image_clean = image[:orig_h, :orig_w]

heatmap = heatmap / np.maximum(counter, 1)
heatmap = 1.0 - heatmap
heatmap_uint8 = (heatmap * 255).astype(np.uint8)
heatmap_blurred = cv2.GaussianBlur(heatmap_uint8, (151, 151), 0)
heatmap_color = cv2.applyColorMap(heatmap_blurred, cv2.COLORMAP_JET)

overlay = cv2.addWeighted(image_clean, 0.6, heatmap_color, 0.4, 0)

# NEW VERSION.

overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots(figsize=(14, 10))
cax = ax.imshow(overlay_rgb)

sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
cbar.set_label(
    "Współczynnik zagrożenia (0.0 = Optymalnie, 1.0 = Krytycznie)",
    rotation=270,
    labelpad=20,
    fontsize=12,
)

ax.axis("off")
plt.tight_layout()

if args.aircraft == "c152":
    plt.savefig(
        f"data/heatmaps/landing_heatmap_c152_{args.image}.png",
        dpi=300,
        bbox_inches="tight",
    )
else:
    plt.savefig(
        f"data/heatmaps/landing_heatmap_b737_{args.image}.png",
        dpi=300,
        bbox_inches="tight",
    )

# OLDER VERSION (without legend) leaving it cus I don't know which one to use for the thesis.

# cv2.imwrite("landing_heatmap.png", overlay)
# print("Result saved as 'landing_heatmap.png'.")

# cv2.imshow("Landing Heatmap", overlay)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
