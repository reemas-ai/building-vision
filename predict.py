import torch
import torch.nn as nn
from torchvision.models import resnet50
from PIL import Image
from torchvision import transforms
import argparse
from pathlib import Path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet50(weights=None)
model.fc = nn.Linear(2048, 2)

model_path = "crack_model.pth"

state_dict = torch.load(
    model_path,
    map_location=device,
    weights_only=True
)

model.load_state_dict(state_dict)
model = model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class_names = ["No Crack Detected", "Crack Detected"]


def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")

    image = transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        predicted_class = output.argmax(dim=1).item()

    return class_names[predicted_class]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to the image")
    args = parser.parse_args()

    if not Path(args.image_path).is_file():
        parser.error("Image file does not exist.")

    prediction = predict_image(args.image_path)
    print("Prediction:", prediction)