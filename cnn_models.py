import torch.nn


class CNN2Layers(torch.nn.Module):

    def __init__(self, in_channels, feature_channels, kernel_size, stride, padding, dropout, batch_size):
        super(CNN2Layers, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=in_channels, out_channels=feature_channels, kernel_size=kernel_size,
                            stride=stride, padding=padding),
            torch.nn.ELU(),
            torch.nn.Dropout(dropout),

            torch.nn.Conv1d(in_channels=feature_channels, out_channels=int(feature_channels/2), kernel_size=kernel_size,
                            stride=stride, padding=padding),
            torch.nn.ELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(in_channels = int(feature_channels/2), out_channels=1, kernel_size=31,
                            stride = 1, padding= 0 )
        )
        
        # self.fc = torch.nn.Linear(int(feature_channels/2)*31*batch_size, batch_size)
            

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        # x = torch.flatten(x)
        # print(x.shape)
        # x = self.fc(x)
        # return x
        return torch.squeeze(x)


class Logistic_Reg_model(torch.nn.Module):
    def __init__(self,no_input_features=10 ):
        super(Logistic_Reg_model,self).__init__()
        self.layer1=torch.nn.Linear(no_input_features,64)
        self.layer2=torch.nn.Linear(64,1)
        # self.subset_models = subset_models
    def forward(self,x):
        # print(x.shape)
        y_predicted=self.layer1(x)
        y_predicted=(self.layer2(y_predicted))
        return y_predicted