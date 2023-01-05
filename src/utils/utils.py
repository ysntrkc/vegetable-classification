import os
import json

os.environ["KAGGLE_USERNAME"] = json.load(open("kaggle.json"))["username"]
os.environ["KAGGLE_KEY"] = json.load(open("kaggle.json"))["key"]

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from kaggle.api.kaggle_api_extended import KaggleApi


def download_data():
    if not os.path.exists("../data"):
        os.mkdir("../data")

    if not os.path.exists("../data/Vegetable Images"):
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(
            "praveengovi/coronahack-chest-xraydataset", path="../data", unzip=True
        )


def get_data_loaders(batch_size):
    train_dir = "../data/Vegetable Images/train"
    val_dir = "../data/Vegetable Images/validation"
    test_dir = "../data/Vegetable Images/test"

    train_transform = transforms.Compose(
        [
            transforms.RandomRotation(12),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomResizedCrop(224),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.12, hue=0.12
            ),
            transforms.RandomAffine(
                degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=20
            ),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    test_transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=test_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader
