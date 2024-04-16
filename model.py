import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)  # Input is 1x28x28, output is 6x28x28
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)            # Input is 6x14x14, output is 16x10x10
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)          # Input is 16x5x5, output is 120x1x1
        self.fc1 = nn.Linear(120, 84)                           # Fully connected layers
        self.fc2 = nn.Linear(84, 10)                            # Output layer (10 classes for MNIST)

    def forward(self, img):
        img = F.max_pool2d(F.relu(self.conv1(img)), (2, 2))  # First subsampling
        img = F.max_pool2d(F.relu(self.conv2(img)), (2, 2))  # Second subsampling
        img = F.relu(self.conv3(img))
        img = img.view(-1, self.num_flat_features(img))
        img = F.relu(self.fc1(img))
        img = self.fc2(img)
        return img

    def num_flat_features(self, img):
        size = img.size()[1:]  # All dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class CustomMLP(nn.Module):
    """ Your custom MLP model

        - Note that the number of model parameters should be about the same
          with LeNet-5
    """
    def __init__(self):
        super(CustomMLP, self).__init__()
        input_size = 28*28  # MNIST images are 28x28 pixels
        hidden1_size = 70  # First hidden layer size
        hidden2_size = 55   # Second hidden layer size
        hidden3_size = 28   # Third hidden layer size
        output_size = 10    # MNIST has 10 classes (0-9)

        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2_size, hidden3_size)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden3_size, output_size)

    def forward(self, img):
        img = img.view(-1, 28*28)  # Flatten the image
        img = self.relu1(self.fc1(img))
        img = self.relu2(self.fc2(img))
        img = self.relu3(self.fc3(img))
        img = self.fc4(img)
        return img


class RegularizedLenet5(nn.Module):
    def __init__(self):
        super(RegularizedLenet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(6)  # 배치 정규화 적용
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)  # 배치 정규화 적용
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        self.fc1 = nn.Linear(120, 84)
        self.dropout1 = nn.Dropout(0.5)  # 드롭아웃 적용
        self.fc2 = nn.Linear(84, 10)

    def forward(self, img):
        img = F.max_pool2d(F.relu(self.bn1(self.conv1(img))), (2, 2))
        img = F.max_pool2d(F.relu(self.bn2(self.conv2(img))), (2, 2))
        img = F.relu(self.conv3(img))
        img = img.view(-1, self.num_flat_features(img))
        img = F.relu(self.fc1(img))
        img = self.dropout1(img)  # 드롭아웃 적용
        img = self.fc2(img)
        return img

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features