import os
import json
import glob
import h5py
import torch

os.environ["KAGGLE_USERNAME"] = json.load(open("kaggle.json"))["username"]
os.environ["KAGGLE_KEY"] = json.load(open("kaggle.json"))["key"]

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from kaggle.api.kaggle_api_extended import KaggleApi


def evaluate_model(model, test_dataloader, loss_fn, device):
    model.eval()
    with torch.no_grad():
        test_loss = 0
        test_acc = 0
        for x_batch, y_batch in test_dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            test_loss += loss.item() * y_batch.size(0)
            is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
            test_acc += torch.sum(is_correct).item()
        test_loss /= len(test_dataloader.dataset)
        test_acc /= len(test_dataloader.dataset)
    return test_loss, test_acc


def get_num_of_models(model_name):
    return len(glob.glob(f"../models/{model_name}/*"))


def create_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def save_results(args, results):
    create_dir_if_not_exists("../results")
    create_dir_if_not_exists(f"../results/{args.model}")

    filename = f"../results/{args.model}/{args.model}_bs{args.batch_size}_lr{args.learning_rate}_epochs{args.epochs}.h5"

    with h5py.File(filename, "w") as f:
        for key, value in results.items():
            f.create_dataset(key, data=value)


def download_data():
    create_dir_if_not_exists("../data")

    if not os.path.exists("../data/Vegetable Images"):
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(
            "misrakahmed/vegetable-image-dataset", path="../data", unzip=True
        )

    create_dir_if_not_exists("../models")


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
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=test_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_dataloader, val_dataloader, test_dataloader
