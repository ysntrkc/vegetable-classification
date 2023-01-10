import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_classes=15):
        super(CNN, self).__init__()

        # Using Sequential container
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(37 * 37 * 64, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 15),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
