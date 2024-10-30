import torch
import numpy as np
import torch.utils.data.dataloader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn

batch_size = 100
input_size = 96 * 96 # 96x96
learning_rate = 0.001
epochs = 2
classes = ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')
output_size = len(classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1) process data

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

train_data = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transform)
test_data = torchvision.datasets.STL10(root='./data', split='test', transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

# images, labels = next(iter(train_loader))
# print(images)
# print(labels)

# print('train_loader', train_data)
kernel_size1 = 5 #filter size
# kernel_size2 = 5
class ConvFerNet(nn.Module):
    def __init__(self):
        super(ConvFerNet, self).__init__()
        # conv output = ((W - F) + 2P)/S + 1 F=filter_size, W=inputSize, P=padding, S=stride
        self.convL1 = torch.nn.Conv2d(3, 6, kernel_size1)
        self.pool = torch.nn.MaxPool2d(2, 2) #cuts inputs in half
        self.convL2 = torch.nn.Conv2d(6, 16, kernel_size1)
        self.linL1 = torch.nn.Linear(16 * 21 * 21, 1200)
        self.linL2 = torch.nn.Linear(1200, 400)
        self.linL3 = torch.nn.Linear(400, 84)
        self.linL4 = torch.nn.Linear(84, output_size)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.convL1(x)))
        x = self.pool(self.relu(self.convL2(x)))
        x = x.view(-1, 16 * 21 * 21)
        x = self.relu(self.linL1(x))
        x = self.relu(self.linL2(x))
        x = self.relu(self.linL3(x))
        x = self.linL4(x)
        return x

model = ConvFerNet().to(device)

# loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
total_steps = len(train_loader)
# train model
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        # images, labels = train_loader[i]
        images = images.to(device)
        labels = labels.to(device)
        output = model(images) # forward pass
        loss = criterion(output, labels) #calculate loss

        optimizer.zero_grad() #zero out last gradient
        loss.backward() # calculate descent gradient for each weight
        optimizer.step() # update weights

        if (i + 1) % 1000 == 0:
            print(f'step: {i}/{total_steps}, loss: {loss.item():.4f}')



