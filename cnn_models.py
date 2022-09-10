import torch.nn


class CNN2Layers(torch.nn.Module):

    def __init__(self, in_channels, feature_channels, kernel_size, stride, padding, dropout):
        super(CNN2Layers, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=in_channels, out_channels=feature_channels, kernel_size=kernel_size,
                            stride=stride, padding=padding),
            torch.nn.ELU(),
            torch.nn.Dropout(dropout),

            torch.nn.Conv2d(in_channels=feature_channels, out_channels=feature_channels/2, kernel_size=kernel_size,
                            stride=stride, padding=padding),
            torch.nn.ELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear((feature_channels/2)*31, 1)
            
        )

    def forward(self, x):
        x = self.conv1(x)
        return x
        # return torch.squeeze(x)