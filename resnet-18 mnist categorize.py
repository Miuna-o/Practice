import torch
from torch.optim import Adam
from torchvision.models import resnet18
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

model=resnet18(num_classes=10)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

def train_test_model(model, criterion, optimizer, train_loader, test_loader, epochs=5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
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

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}, Accuracy: {train_accuracy}%')

        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accuracy = 100 * correct / total
        print(f'Test Loss: {test_loss / len(test_loader)}, Test Accuracy: {test_accuracy}%')



if __name__=='__main__':
    train_dataset = MNIST("mnist", train=True, download=True, transform=ToTensor())
    test_dataset = MNIST("mnist", train=False, download=True, transform=ToTensor())

    train = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test = DataLoader(test_dataset, batch_size=64)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    train_test_model(model, criterion, optimizer, train, test, epochs=5)
####
