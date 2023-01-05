import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", "-bs", type=int, default=32, help="batch size")
    parser.add_argument(
        "--epochs", "-e", type=int, default=10, help="number of epochs of training"
    )
    parser.add_argument(
        "--learning_rate", "-lr", type=float, default=0.001, help="learning rate"
    )
    parser.add_argument(
        "--num_classes", "-nc", type=int, default=15, help="number of classes"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="CNN",
        help="model to use",
        choices=["CNN", "ResNet"],
    )

    args = parser.parse_args()
    return args
