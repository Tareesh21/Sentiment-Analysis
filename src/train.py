import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from src.datasets import EmotionDataset
from src.models.cnn import EmotionCNN
from src.models.vgg16_transfer import get_vgg16
import torch.optim as optim
import time
import os

def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=25):
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_acc = 100 * correct / total
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/total:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
    print(f'Best Validation Accuracy: {best_val_acc:.2f}%')

def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def main():
    train_dir = 'data/train'
    val_dir = 'data/val'
    train_labels = ...  
    val_labels = ...    
    face_cascade_path = 'haarcascade_frontalface_default.xml'
    batch_size = 64
    num_epochs = 25
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = EmotionDataset(train_dir, train_labels, transform, face_cascade_path)
    val_dataset = EmotionDataset(val_dir, val_labels, transform, face_cascade_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model = EmotionCNN(num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs)

if __name__ == '__main__':
    main()
