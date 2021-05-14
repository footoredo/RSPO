import torch
import numpy as np


def main():
    width = 10
    samples = 100000
    net = torch.nn.Sequential(torch.nn.Linear(width, width),
                              torch.nn.ReLU(),
                              torch.nn.Linear(width, 1))

    x = torch.randn(samples, width)
    y = net(x)
    print(np.std(y.detach().numpy()))

    for i in range(5):
        x_0 = torch.randn(1, width - 1)
        x_1 = torch.randn(samples, 1)
        x = torch.cat((x_1, x_0.repeat((samples, 1))), dim=1)
        y = net(x)
        print(np.std(y.detach().numpy()))

if __name__ == "__main__":
    main()
