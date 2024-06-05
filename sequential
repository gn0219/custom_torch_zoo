
# SimpleDNN
class SimpleDNN(nn.Module):
    def __init__(self, target_N=20, n_of_features=3):
        super(SimpleDNN, self).__init__()
        self.flatten = nn.Flatten()
        self.seq_module = nn.Sequential(
            nn.BatchNorm1d(target_N*n_of_features),
            nn.Linear(target_N*n_of_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        x = self.flatten(x)
        x = self.seq_module(x)
        return x

# Baseline(DNN) See section 3. DNN Model Building
class Baseline(nn.Module):
    def __init__(self, target_N=20, n_of_features=3):
        super(Baseline, self).__init__()
        # Input size (after flattening): 20 * 3 = 60
        # Write your code here
        self.flatten = nn.Flatten()
        self.seq_module = nn.Sequential(
            nn.BatchNorm1d(target_N*n_of_features),
            nn.Linear(target_N*n_of_features, 500),
            nn.ELU(),
            nn.BatchNorm1d(500),
            nn.Linear(500, 250),
            nn.ELU(),
            nn.BatchNorm1d(250),
            nn.Linear(250, 125),
            nn.ELU(),
            nn.BatchNorm1d(125),
            nn.Linear(125, 50),
            nn.BatchNorm1d(50),
            nn.Linear(50, 2),
        )
        self.initialize_weights()

    def forward(self, x):
        # Write your code here
        x = self.flatten(x)
        x = self.seq_module(x)
        return x

    def initialize_weights(self):
      for m in self.modules():
        if isinstance(m, nn.Linear):
          nn.init.kaiming_normal_(m.weight)
          nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm1d):
          nn.init.constant_(m.weight.data, 1)
          nn.init.constant_(m.bias.data, 0)

# ConvLSTM model
class ConvLSTM(nn.Module):
    def __init__(self, input_channels=20):
        super(ConvLSTM, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        
        self.lstm = nn.LSTM(input_size=128, hidden_size=30, num_layers=2, bias=True, bidirectional=False, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(30, 2)

    def forward(self, x):
        x = self.conv_layers(x)  # (batch_size, 128, seq_length)
        x = x.permute(0, 2, 1)  # Change to (batch_size, seq_length, 128)
        x, (hn, cn) = self.lstm(x)  # LSTM output: (batch_size, seq_length, hidden_size)
        x = x[:, -1, :]  # Take the last output of the sequence (batch_size, hidden_size)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

# Multibranch CNN-BiLSTM two branches
class CNNBiLSTM2b(nn.Module):
    def __init__(self, target_N=20, n_of_features=3, filter_sizes=[3, 7]):
        super(CNNBiLSTM2b, self).__init__()
        self.conv_branch1 = self._make_layers(input_channels=target_N, filter_size=filter_sizes[0])
        self.conv_branch2 = self._make_layers(input_channels=target_N, filter_size=filter_sizes[1])
        self.bilstm1 = nn.LSTM(input_size=(n_of_features//2+1)*2*32, hidden_size=64, num_layers=1, bias=True, bidirectional=True, batch_first=True)
        # self.bilstm1 = nn.LSTM(input_size=32*2, hidden_size=64, num_layers=1, bias=True, bidirectional=True, batch_first=True)
        self.bilstm2 = nn.LSTM(input_size=64*2, hidden_size=32, num_layers=1, bias=True, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        # self.classifier = nn.Sequential(
        #     nn.Linear(32*2, 16),
        #     nn.Linear(16, 2),
        # )
        self.classifier = nn.Linear(32*2, 2)
    def _make_layers(self, input_channels, filter_size):
        layers = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=filter_size, stride=1, padding=(filter_size-1)//2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=filter_size, stride=1, padding=(filter_size-1)//2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout1d(0.5),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Flatten()
        )
        return layers
    
    def forward(self, x):
        x1 = self.conv_branch1(x)
        x2 = self.conv_branch2(x)
        x = torch.cat([x1, x2], dim=1)
        x = x.unsqueeze(1)
        x, _ = self.bilstm1(x)
        x, _ = self.bilstm2(x)
        x = x.squeeze()
        x = self.dropout(x)
        x = self.classifier(x)
        return x
    
# Multibranch CNN-BiLSTM three branches
class CNNBiLSTM3b(nn.Module):
    def __init__(self, target_N=20, n_of_features=3, filter_sizes=[3, 5, 7]):
        super(CNNBiLSTM3b, self).__init__()
        self.conv_branch1 = self._make_layers(input_channels=target_N, filter_size=filter_sizes[0])
        self.conv_branch2 = self._make_layers(input_channels=target_N, filter_size=filter_sizes[1])
        self.conv_branch3 = self._make_layers(input_channels=target_N, filter_size=filter_sizes[2])
        # self.bilstm1 = nn.LSTM(input_size=32*3*2, hidden_size=64, num_layers=1, bias=True, bidirectional=True, batch_first=True)
        self.bilstm1 = nn.LSTM(input_size=(n_of_features//2+1)*3*32, hidden_size=64, num_layers=1, bias=True, bidirectional=True, batch_first=True)
        self.bilstm2 = nn.LSTM(input_size=64*2, hidden_size=32, num_layers=1, bias=True, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        # self.classifier = nn.Sequential(
        #     nn.Linear(32*2, 16),
        #     nn.Linear(16, 2),
        # )
        self.classifier = nn.Linear(32*2, 2)
    def _make_layers(self, input_channels, filter_size):
        layers = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=filter_size, stride=1, padding=(filter_size-1)//2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=filter_size, stride=1, padding=(filter_size-1)//2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout1d(0.5),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Flatten()
        )
        return layers
    
    def forward(self, x):
        x1 = self.conv_branch1(x)
        x2 = self.conv_branch2(x)
        x3 = self.conv_branch3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = x.unsqueeze(1)
        x, _ = self.bilstm1(x)
        x, _ = self.bilstm2(x)
        x = x.squeeze()
        x = self.dropout(x)
        x = self.classifier(x)
        return x
