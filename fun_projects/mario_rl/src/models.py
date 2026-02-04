import torch.nn as nn
from torchinfo import summary


def default_cnn(in_channels, num_actions):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3136, 512),
        nn.ReLU(),
        nn.Linear(512, num_actions),
    )


def simple_cnn(in_channels, num_actions):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=5, stride=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
        nn.ReLU(),
        nn.Flatten(),
        nn.LazyLinear(512),
        nn.ReLU(),
        nn.Linear(512, num_actions),
    )


if __name__ == "__main__":
    input_size = (1, 4, 128, 128)
    model = simple_cnn(input_size[1], 2)
    summary(model, input_size=input_size)
