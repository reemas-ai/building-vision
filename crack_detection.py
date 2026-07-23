import os
from torchvision import transforms
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch          
import torchvision    
from PIL import Image 
import matplotlib.pyplot as plt  
import random
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
#--------------------------------------------------------------------------
# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
#--------------------------------------------------------------------------
#add the path 
import kagglehub
from pathlib import Path
base_path = Path(kagglehub.dataset_download("arunrk7/surface-crack-detection"))
negative_image_path = base_path / 'Negative'
positive_image_path = base_path / 'Positive'
#load 200 image 
image_negative=list(negative_image_path.glob("*"))[:2000]
image_positive=list(positive_image_path.glob("*"))[:2000]
print(f"Negative: {len(image_negative)}")
print(f"Positive: {len(image_positive)}")
#--------------------------------------------------------------------------
#make a tensor
to_tensor = torchvision.transforms.ToTensor()
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                     std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                     std=[0.229, 0.224, 0.225])
])
#--------------------------------------------------------------------------
#function : conver the image to tensor 
def image_to_tensor(images, label, transform):
    image_tensor = []
    for image in images:
        img = Image.open(image).convert("RGB")
        img = transform(img)
        image_tensor.append((img, label))
    return image_tensor
#--------------------------------------------------------------------------
#The Dataset after conver 
negative_data = image_to_tensor(image_negative, 0, transform_train)
positive_data = image_to_tensor(image_positive, 1, transform_train)
negative_test = image_to_tensor(image_negative, 0, transform_test)
positive_test = image_to_tensor(image_positive, 1, transform_test)
dataset_train = negative_data + positive_data
dataset_test = negative_test + positive_test
random.shuffle(dataset_train)
random.shuffle(dataset_test)

train_data = dataset_train
test_data = dataset_test
print(f"Total: {len(dataset_train) + len(dataset_test)}")
#--------------------------------------------------------------------------
#make training and testing data

data_train_loader=DataLoader(train_data,batch_size=32,shuffle=True)
data_test_loader=DataLoader(test_data,batch_size=32,shuffle=False)
#--------------------------------------------------------------------------
#Download the model
model = resnet50(weights=ResNet50_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False
num_classes=2
model.fc = nn.Linear(in_features=2048, out_features=num_classes)
model = model.to(device)
print(model.fc)
#--------------------------------------------------------------------------

criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  

num_epochs = 10

for epochs in range(num_epochs):
    for images, labels in data_train_loader:
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        loss = criterion(output, labels)    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
    print(f"Epoch {epochs+1}/{num_epochs} - Loss: {loss.item():.4f}")

correct = 0
total = 0
#--------------------------------------------------------------------------
#test
with torch.no_grad():
    for images, labels in data_test_loader:
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total * 100
print(f"Accuracy: {accuracy:.2f}%")
#--------------------------------------------------------------------------
#save the model
torch.save(model.state_dict(), 'crack_model.pth')
print("Model saved!")
#--------------------------------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                     std=[0.229, 0.224, 0.225])
                     ])
model.eval()
with torch.no_grad():
    image_new = Image.open("x.png").convert("RGB")
    image_new = transform(image_new)
    image_new = image_new.unsqueeze(0).to(device)
    output_new = model(image_new)
    _, predicted = torch.max(output_new, 1)
    if predicted.item() == 1:
        print("Crack Detected! ⚠️")
    else:
        print("No Crack - Safe ✅")

