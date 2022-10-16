# import torch.nn 
import torch

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

class LSTM_base(torch.nn.ModuleList):

    def __init__(self, config):
        super(LSTM_base, self).__init__()
        
        self.batch_size = config["batch_size"]
        self.hidden_dim = config["hidden_dim"]
        self.LSTM_layers = config["lstm_layers"]
        self.input_size = config["emdedding_len"] # embedding dimention
        
        
        self.dropout = torch.nn.Dropout(config["dropout_ratio"])
        # self.embedding = nn.Embedding(self.input_size, self.hidden_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_dim, num_layers=self.LSTM_layers, batch_first=True)
        self.fc1 = torch.nn.Linear(in_features=self.hidden_dim, out_features=64)
        self.fc2 = torch.nn.Linear(64, 1)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, x):
    
        h = torch.zeros((self.LSTM_layers, x.size(0), self.hidden_dim)).to(self.device)
        c = torch.zeros((self.LSTM_layers, x.size(0), self.hidden_dim)).to(self.device)
        
        torch.nn.init.xavier_normal_(h)
        torch.nn.init.xavier_normal_(c)

        out = x
    # print(x.shape)
        out, (hidden, cell) = self.lstm(out, (h,c))
        out = self.dropout(out)
        out = torch.nn.functional.relu(self.fc1(out[:,-1,:]))
        out = self.dropout(out)
        out = (self.fc2(out))

        return out


