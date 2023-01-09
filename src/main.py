import time
import torch
import torch.nn as nn
from torchvision import models

from utils.utils import (
    get_data_loaders,
    download_data,
    create_dir_if_not_exists,
    get_num_of_models,
    evaluate_model,
    save_results,
)
from utils.options import args_parser
from model.CNN import CNN


def train(
    args,
    model,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    lr_scheduler,
    optimizer,
    loss_fn,
):
    create_dir_if_not_exists(f"../models/{args.model}")
    model_num = get_num_of_models(args.model)
    model.to(args.device)
    max_acc = 0
    torch.save(
        model.state_dict(), f"../models/{args.model}/{args.model}_{model_num}.pt"
    )
    train_loss, train_acc = [0] * args.epochs, [0] * args.epochs
    val_loss, val_acc = [0] * args.epochs, [0] * args.epochs
    test_loss, test_acc = [0] * args.epochs, [0] * args.epochs
    count = 0
    for epoch in range(args.epochs):
        if count == 10:
            break
        count += 1
        start_time = time.time()
        model.train()
        for x_batch, y_batch in train_dataloader:
            x_batch, y_batch = x_batch.to(args.device), y_batch.to(args.device)
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
                x_batch, y_batch = x_batch.to(args.device), y_batch.to(args.device)
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                val_loss[epoch] += loss.item() * y_batch.size(0)
                is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
                val_acc[epoch] += torch.sum(is_correct).item()
        val_loss[epoch] /= len(val_dataloader.dataset)
        val_acc[epoch] /= len(val_dataloader.dataset)
        lr_scheduler.step()
        if val_acc[epoch] > max_acc:
            count = 0
            max_acc = val_acc[epoch]
            torch.save(
                model.state_dict(),
                f"../models/{args.model}/{args.model}_{model_num}.pt",
            )
        test_loss[epoch], test_acc[epoch] = evaluate_model(
            model, test_dataloader, loss_fn, args.device
        )
        end_time = time.time()
        print(
            f"Epoch {epoch+1:>2}/{args.epochs} | Train Loss: {train_loss[epoch]:.4f} | Train Acc: {train_acc[epoch]:.4f}",
            end=" | ",
        )
        print(
            f"Val Loss: {val_loss[epoch]:.4f} | Val Acc: {val_acc[epoch]:.4f}",
            end=" | ",
        )
        print(
            f"Test Loss: {test_loss[epoch]:.4f} | Test Acc: {test_acc[epoch]:.4f}",
            end=" | ",
        )
        print(f"Time taken: {end_time - start_time:.2f} seconds")

    return {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }


def main():
    args = args_parser()
    download_data()

    args.device = torch.device(
        f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
    )

    train_dataloader, val_dataloader, test_dataloader = get_data_loaders(
        args.batch_size
    )

    if (args.model).lower() == "resnet":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        features = list(model.fc.parameters())[0].shape[1]
        model.fc = nn.Linear(features, args.num_classes)
    elif (args.model).lower() == "cnn":
        model = CNN(args.num_classes)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=7, gamma=0.1
    )

    hist = train(
        args,
        model,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        exp_lr_scheduler,
        optimizer,
        loss_fn,
    )

    save_results(args, hist)


if __name__ == "__main__":
    main()
