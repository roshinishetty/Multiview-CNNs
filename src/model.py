import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        """
        
        DEFINE YOUR NETWORK HERE
        
        """

        self.conv1 = nn.Conv2d(1, 8, 7, stride=1, padding=0, bias=True)
        self.layernorm1 = nn.LayerNorm(normalized_shape=[8, 106, 106])
        self.leakyReLU = nn.LeakyReLU(negative_slope=0.01)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.depthwise_conv1 = nn.Conv2d(in_channels=8, out_channels=8, groups=8, kernel_size=7, bias=True, stride=2)
        self.layernorm2 = nn.LayerNorm(normalized_shape=[8, 24, 24])
        self.pointwise_conv1 = nn.Conv2d(8, 16, 1, bias=True)
        self.depthwise_conv2 = nn.Conv2d(in_channels=16, out_channels=16, groups=16, kernel_size=7, bias=True)
        self.layernorm3 = nn.LayerNorm(normalized_shape=[16, 6, 6])
        self.pointwise_conv2 = nn.Conv2d(16, 32, 1, bias=True)
        self.fully_connected = nn.Conv2d(in_channels=32*3*3, out_channels=10, kernel_size=1, bias=True)

        # Initialize weights
        torch.nn.init.kaiming_uniform_(self.conv1.weight)
        torch.nn.init.kaiming_uniform_(self.depthwise_conv1.weight)
        torch.nn.init.kaiming_uniform_(self.depthwise_conv2.weight)
        torch.nn.init.xavier_uniform_(self.pointwise_conv1.weight)
        torch.nn.init.xavier_uniform_(self.pointwise_conv2.weight)
        torch.nn.init.xavier_uniform_(self.fully_connected.weight)

        # Initialize bias to 0
        torch.nn.init.zeros_(self.conv1.bias)
        torch.nn.init.zeros_(self.depthwise_conv1.bias)
        torch.nn.init.zeros_(self.depthwise_conv2.bias)
        torch.nn.init.zeros_(self.pointwise_conv1.bias)
        torch.nn.init.zeros_(self.pointwise_conv2.bias)
        torch.nn.init.zeros_(self.fully_connected.bias)

        
    def forward(self, x):
        """

        DEFINE YOUR FORWARD PASS HERE

        """

        out = self.conv1(x)
        out = self.layernorm1(out)
        out = self.leakyReLU(out)
        out = self.maxpool(out)

        out = self.depthwise_conv1(out)
        out = self.layernorm2(out)
        out = self.leakyReLU(out)
        out = self.maxpool(out)
        out = self.pointwise_conv1(out)

        out = self.depthwise_conv2(out)
        out = self.layernorm3(out)
        out = self.leakyReLU(out)
        out = self.maxpool(out)
        out = self.pointwise_conv2(out)

        out = torch.reshape(input=out, shape=(-1, 32*3*3, 1, 1))
        out = self.fully_connected(out)
        
        return out
