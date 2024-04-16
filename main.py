import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import CustomMLP,LeNet5,RegularizedLenet5
from dataset import MNIST

def train(model, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
        acc: accuracy
    """
    model.train()
    trn_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trn_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        trn_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    trn_loss = trn_loss / len(trn_loader)
    acc = 100. * correct / total
    return trn_loss, acc

def test(model, tst_loader, device, criterion):
    """ Test function

    Args:
        model: network
        tst_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        tst_loss: average loss value
        acc: accuracy
    """
    model.eval()
    tst_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tst_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            tst_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    tst_loss = tst_loss / len(tst_loader)
    acc = 100. * correct / total
    return tst_loss, acc

def main():
    """ Main function

        Here, you should instantiate
        1) Dataset objects for training and test datasets
        2) DataLoaders for training and testing
        3) model
        4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
        5) cost function: use torch.nn.CrossEntropyLoss

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataset instances and dataloaders
    train_data_dir = '/dataset/train.tar'
    test_data_dir = '/dataset/test.tar'
    train_dataset = MNIST(data_dir=train_data_dir)
    test_dataset = MNIST(data_dir=test_data_dir)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Model, Loss, Optimizer
    models = {'LeNet5': LeNet5(), 'RegularizedLenet5': RegularizedLenet5(), 'CustomMLP': CustomMLP()}
    criterion = nn.CrossEntropyLoss()
    stats = {}

    for name, model in models.items():
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        trn_loss_list, tst_loss_list, trn_acc_list, tst_acc_list = [], [], [], []

        for epoch in range(10):  # run for 10 epochs
            trn_loss, trn_acc = train(model, train_loader, device, criterion, optimizer)
            tst_loss, tst_acc = test(model, test_loader, device, criterion)

            trn_loss_list.append(trn_loss)
            tst_loss_list.append(tst_loss)
            trn_acc_list.append(trn_acc)
            tst_acc_list.append(tst_acc)

            print(f'Epoch {epoch+1}, {name} Train Loss: {trn_loss:.4f}, Accuracy: {trn_acc:.2f}%')
            print(f'Epoch {epoch+1}, {name} Test Loss: {tst_loss:.4f}, Accuracy: {tst_acc:.2f}%')

        stats[name] = (trn_loss_list, tst_loss_list, trn_acc_list, tst_acc_list)

    # Plotting
    for name, (trn_loss_list, tst_loss_list, trn_acc_list, tst_acc_list) in stats.items():
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(trn_loss_list, label='Train Loss')
        plt.plot(tst_loss_list, label='Test Loss')
        plt.title(f'Loss over Epochs ({name})')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(trn_acc_list, label='Train Accuracy')
        plt.plot(tst_acc_list, label='Test Accuracy')
        plt.title(f'Accuracy over Epochs ({name})')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    main()