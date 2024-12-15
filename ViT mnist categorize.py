import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class ViT(nn.Module):
    def __init__(self, num_classes=10):
        super(ViT, self).__init__()
        self.patch_embed = nn.Linear(784, 128)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 128))
        self.pos_embed = nn.Parameter(torch.randn(1, 2, 128))
        self.transformer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256)
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, 28, 28)
        x = self.patch_embed(x.flatten(1))  # (B, 784) -> (B, 128)
        x = x.unsqueeze(1)  # (B, 128) -> (B, 1, 128)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (1, 1, 128) -> (B, 1, 128)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1, 128) + (B, 1, 128) -> (B, 2, 128)
        x = x + self.pos_embed
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.head(x[:, 0])
        return x

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.flatten(x))
])


train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViT().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print(f'Epoch :{epoch+1}, Step {i+1}, Loss: {loss.item():.4f}')

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total}%')
###