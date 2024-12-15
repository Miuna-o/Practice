import torch
import tensorflow as tf
from torch.nn import Module
from torch import nn
from torch import optim
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

class Lenet(Module):
    def __init__(self):
        super(Lenet,self).__init__()

        self.conv1=nn.Sequential(
            nn.Conv2d(1,6,5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv2=nn.Sequential(
            nn.Conv2d(6,16,5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1=nn.Sequential(
            nn.Linear(256,120),
            nn.ReLU()
        )

        self.fc2=nn.Sequential(
            nn.Linear(120,84),
            nn.ReLU()
        )

        self.fc3=nn.Linear(84,10)

    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(-1,256)
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)
        return x

if __name__=='__main__':
    transform=transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    train_dataset=datasets.ImageFolder('root=./mnist_image/train',transform=transform)
    train_loader=DataLoader(train_dataset,batch_size=64,shuffle=True)
    model=Lenet()
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters())

    for epoch in range(10):
        for batch_idx,(data,label) in enumerate(train_loader):
            output=model(data)
            loss=criterion(output,label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if batch_idx%500==0:
                print("loss:",loss.item())
###