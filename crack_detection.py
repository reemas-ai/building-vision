import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import random
from PIL import Image 
from torchvision import transforms
import torch
import kagglehub
from pathlib import Path          
import torchvision    
import torch.nn as nn  
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50, ResNet50_Weights
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
#--------------------------------------------------------------------------
# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
#--------------------------------------------------------------------------
#add the path 
base_path = Path(kagglehub.dataset_download("arunrk7/surface-crack-detection"))
negative_image_path = base_path / 'Negative'
positive_image_path = base_path / 'Positive'
image_negative=list(negative_image_path.glob("*"))[:2000]
image_positive=list(positive_image_path.glob("*"))[:2000]
print(f"Negative: {len(image_negative)}")
print(f"Positive: {len(image_positive)}")
#--------------------------------------------------------------------------
#image processing function
def image_processing(image_path):
    
    image = cv2.imread(str(image_path))  

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return Image.fromarray(image_rgb)
#--------------------------------------------------------------------------
#the data 
all_data = [(img, 0) for img in image_negative] + [(img, 1) for img in image_positive]
random.shuffle(all_data)
split_idx = int(len(all_data) * 0.8)
train_pairs = all_data[:split_idx]
test_pairs = all_data[split_idx:]

print(f"Train samples: {len(train_pairs)}, Test samples: {len(test_pairs)}")
#--------------------------------------------------------------------------
#make a tensor
to_tensor = torchvision.transforms.ToTensor()
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
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
class CrackDataset(Dataset):
    def __init__(self, image_pairs, transform=None):
        self.image_pairs = image_pairs
        self.transform = transform

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img_path, label = self.image_pairs[idx]
        image = image_processing(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label
#--------------------------------------------------------------------------
#The Dataset after conver 
data_train = DataLoader(CrackDataset(train_pairs, transform_train), batch_size=32, shuffle=True)
data_test= DataLoader(CrackDataset(test_pairs, transform_test), batch_size=32, shuffle=False)

print(f"Total: {len(train_pairs) + len(test_pairs)}")
#--------------------------------------------------------------------------
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
losses = []
for epochs in range(num_epochs):
    for images, labels in data_train:
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        loss = criterion(output, labels)    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
    losses.append(loss.item())
    print(f"Epoch {epochs+1}/{num_epochs} - Loss: {loss.item():.4f}")
    
plt.plot(range(1, num_epochs + 1), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.grid()
plt.savefig('training_loss.png')
plt.show()
correct = 0
total = 0
#--------------------------------------------------------------------------
#test
labels_all=[]
predicted_all=[]
with torch.no_grad():
    for images, labels in data_test:
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        labels_all.extend(labels.cpu().numpy())
        predicted_all.extend(predicted.cpu().numpy())
    
accuracy = correct / total * 100
print(f"Accuracy: {accuracy:.2f}%")
cm=confusion_matrix(labels_all, predicted_all)
print("Confusion Matrix:",cm)
print(classification_report(labels_all, predicted_all, 
      target_names=['No Crack', 'Crack']))
#--------------------------------------------------------------------------
#save the model
torch.save(model.state_dict(), 'crack_model.pth')
print("Model saved!")
#--------------------------------------------------------------------------
def predict_image(image_path):
    model.eval()
    with torch.no_grad():
        image = image_processing(image_path)
        img_tensor = transform_test(image).unsqueeze(0).to(device)
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        
        return "Crack Detected! " if predicted.item() == 1 else "No Crack - Safe "
print(predict_image("19999.jpg"))
print(predict_image("y.png"))
print(predict_image("19995.jpg"))
print(predict_image("x.png"))

