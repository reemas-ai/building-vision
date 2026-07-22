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
#make a tensor
to_tensor = torchvision.transforms.ToTensor()
transform_train = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor()
])
#function : conver the image to tensor 
def image_to_tensor(images, label, transform):
    image_tensor = []
    for image in images:
        img = Image.open(image).convert("RGB")
        img = transform(img)
        image_tensor.append((img, label))
    return image_tensor
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

#make training and testing data

data_train_loader=DataLoader(train_data,batch_size=32,shuffle=True)
data_test_loader=DataLoader(test_data,batch_size=32,shuffle=False)




class CrackDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3, 16,kernel_size=3, padding=1)
        self.pool1=nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2=nn.Conv2d(16,32 ,kernel_size=3, padding=1)
        self.pool2=nn.MaxPool2d(kernel_size=2, stride=2)
        self.liner1=nn.Linear(in_features=100352, out_features=128)
        self.liner2=nn.Linear(in_features=128, out_features=2)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.liner1(x))
        x = self.liner2(x) 
        return x
model = CrackDetector().to(device)
print(model)

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


torch.save(model.state_dict(), 'crack_model.pth')
print("Model saved!")


transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor()
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

