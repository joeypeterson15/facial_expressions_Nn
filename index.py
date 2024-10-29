import torch
import numpy as np
import torch.utils.data.dataloader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import kaggle



batch_size = 100
input_size = 48 * 48 # 48x48
learning_rate = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1) process data

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# train_data = torchvision.datasets.Flowers102(root='./data', split='train', transform=transform, download=True)
train_data = torchvision.datasets.CelebA(root='data/', split='train', download=True, transform=transform)
test_data = torchvision.datasets.Flowers102(root='./data', split='test', transform=transform, download=True)
train_loader = torch.utils.data.dataloader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.dataloader(test_data, batch_size=batch_size, shuffle=False)

print('train_loader', train_loader.size())
classes = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')

class ConvFerNet(torch.nn.Module):
    def __init__(self):
        super(ConvFerNet, self).__init__()
        self.convL1 = torch.nn.Conv2d(1)