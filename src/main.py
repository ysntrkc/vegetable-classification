import torch
import torch.nn as nn
from torchvision import models

from utils.utils import get_data_loaders, download_data
from utils.options import args_parser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    model,
    num_epochs,
    train_dataloader,
    val_dataloader,
    lr_scheduler,
    model_name,
    optimizer,
    loss_fn,
):
    model.to(device)
    max_acc = 0
    torch.save(model.state_dict(), f"../models/{model_name}.pt")
    train_loss = [0] * num_epochs
    train_acc = [0] * num_epochs
    val_loss = [0] * num_epochs
    val_acc = [0] * num_epochs
    for epoch in range(num_epochs):
        model.train()
        count = 0
        for x_batch, y_batch in train_dataloader:
            if count == 50:
                break
            count += 1
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss[epoch] += loss.item() * y_batch.size(0)
            is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
            train_acc[epoch] += torch.sum(is_correct).item()
        train_loss[epoch] /= len(train_dataloader.dataset)
        train_acc[epoch] /= len(train_dataloader.dataset)

        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in val_dataloader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                val_loss[epoch] += loss.item() * y_batch.size(0)
                is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
                val_acc[epoch] += torch.sum(is_correct).item()
        val_loss[epoch] /= len(val_dataloader.dataset)
        val_acc[epoch] /= len(val_dataloader.dataset)
        lr_scheduler.step()
        if val_acc[epoch] > max_acc:
            max_acc = val_acc[epoch]
            torch.save(model.state_dict(), f"../models/{model_name}.pt")
        print(
            f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss[epoch]:.4f} | Train Acc: {train_acc[epoch]:.4f} | Val Loss: {val_loss[epoch]:.4f} | Val Acc: {val_acc[epoch]:.4f}"
        )
    return train_loss, val_loss, train_acc, val_acc


def main():
    args = args_parser()

    train_dataloader, val_dataloader, test_dataloader = get_data_loaders(
        args.batch_size
    )

    if (args.model).lower() == "resnet":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        features = list(model.fc.parameters())[0].shape[1]
        model.fc = nn.Linear(features, args.num_classes)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=7, gamma=0.1
    )

    hist = train(
        model,
        args.epochs,
        train_dataloader,
        val_dataloader,
        exp_lr_scheduler,
        args.model,
        optimizer,
        loss_fn,
    )


if __name__ == "__main__":
    download_data()
    main()
