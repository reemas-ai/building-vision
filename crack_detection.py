import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch          
import torchvision    
from PIL import Image 
import matplotlib.pyplot as plt  
from pathlib import Path 
import random
from torch.utils.data import DataLoader
import torch.nn as nn
#add the path 
negative_image_path = Path('/root/.cache/kagglehub/datasets/arunrk7/surface-crack-detection/versions/1/Negative')
positive_image_path = Path('/root/.cache/kagglehub/datasets/arunrk7/surface-crack-detection/versions/1/Positive')
#load 200 image 
image_negative=list(negative_image_path.glob("*"))[:400]
image_positive=list(positive_image_path.glob("*"))[:400]
print(f"Negative: {len(image_negative)}")
print(f"Positive: {len(image_positive)}")
#make a tensor
to_tensor = torchvision.transforms.ToTensor()
#function : conver the image to tensor 
def image_to_tensor(images,label):
    image_tensor=[]
    for image in images:
       image=Image.open(image)
       image= to_tensor(image)
       image_tensor.append((image,label))
    return image_tensor
#The Dataset after conver 
negative_data = image_to_tensor(image_negative, 0)
positive_data = image_to_tensor(image_positive, 1)
dataset = negative_data + positive_data
random.shuffle(dataset)
print(f"Total: {len(dataset)}")

#make training and testing data

split=int(0.8*(len(dataset)))
train_data=dataset[:split]
test_data=dataset[split:]
print(f"split{split}    train{len(train_data)}  test{len(test_data)}")

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
model = CrackDetector()
print(model)

criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  

num_epochs = 3

for epochs in range(num_epochs):
    for images, labels in data_train_loader:
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
        output = model(images)
        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total * 100
print(f"Accuracy: {accuracy:.2f}%")

from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor()
])
model.eval()
with torch.no_grad():
    image_new = Image.open("x.png").convert("RGB")
    image_new = transform(image_new)
    image_new = image_new.unsqueeze(0)
    output_new = model(image_new)
    _, predicted = torch.max(output_new, 1)
    if predicted.item() == 1:
        print("Crack Detected! ⚠️")
    else:
        print("No Crack - Safe ✅")
