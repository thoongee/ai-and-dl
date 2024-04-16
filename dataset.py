import tarfile
from PIL import Image
import os
import torch
import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class MNIST(Dataset):
    """ MNIST dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        data_dir: directory path containing images

    Note:
        1) Each image should be preprocessed as follows:
            - First, all values should be in a range of [0,1]
            - Substract mean of 0.1307, and divide by std 0.3081
            - These preprocessing can be implemented using torchvision.transforms
        2) Labels can be obtained from filenames: {number}_{label}.png
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_open = tarfile.open(data_dir, 'r')
        self.png_files = [file for file in self.file_open.getmembers() if file.isfile()]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.1307], [0.3081])
        ])

    def __len__(self):
        return len(self.png_files)

    def __getitem__(self, idx):
        member = self.png_files[idx]
        label = int(member.name.split('_')[1].split('.')[0])

        file = self.file_open.extractfile(member)
        image = Image.open(io.BytesIO(file.read())).convert('L')
        image = self.transform(image)
        label = torch.tensor(label)
        return image, label

if __name__ == '__main__':
    # Define paths to the tar files for training and testing datasets
    train_data_dir = '/content/gdrive/MyDrive/AI/train.tar'
    test_data_dir = '/content/gdrive/MyDrive/AI/test.tar'

    # Create dataset instances
    train_dataset = MNIST(data_dir=train_data_dir)
    test_dataset = MNIST(data_dir=test_data_dir)

    # Create DataLoader instances
    train_data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Example of fetching a batch of data
    images, labels = next(iter(train_data_loader))
    print(images.size())  # torch.Size([64, 1, 28, 28])
    print(labels.size())  # torch.Size([64])