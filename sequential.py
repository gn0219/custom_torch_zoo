
# DNN
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.seq_module = nn.Sequential(
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        # self.initialize_weights()

    def forward(self, x):
        x = self.seq_module(x)
        return x

class DNN_medium(nn.Module):
    def __init__(self):
        super(DNN_medium, self).__init__()
        self.classifier = nn.Sequential(
            nn.LazyBatchNorm1d(),
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.classifier(x)
        return x

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

# CNN1d

def make_cnn_layers():
        layers = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.Conv1d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool1d(7)
        )
        return layers

class CNN1d(nn.Module):
    def __init__(self):
        super(CNN1d, self).__init__()
        self.encoder = make_cnn_layers()
        self.classifier = nn.Sequential(
            nn.Linear(448, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)

        x = x.view(x.size(0), -1)  # Reshape to flatten
        x = self.classifier(x)
        return x

class CNN1dCat(nn.Module):
    def __init__(self):
        super(CNN1dCat, self).__init__()
        self.features_c = make_cnn_layers()
        self.features_ms = make_cnn_layers()
        self.features_mfcc = make_cnn_layers()

        self.classifier = nn.Sequential(
            nn.Linear(448 * 3, 64),  # Updated input size based on the concatenated features
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_c = x[:, [i for i in range(12)]].unsqueeze(1)
        x_ms = x[:, [i for i in range(12, 140)]].unsqueeze(1)
        x_mfcc = x[:, [i for i in range(140, 180)]].unsqueeze(1)
        x_c = self.features_c(x_c)
        x_ms = self.features_ms(x_ms)
        x_mfcc = self.features_mfcc(x_mfcc)

        x_c = x_c.view(x_c.size(0), -1)  # Reshape to flatten
        x_ms = x_ms.view(x_ms.size(0), -1)
        x_mfcc = x_mfcc.view(x_mfcc.size(0), -1)
        
        x = torch.cat((x_c, x_ms, x_mfcc), dim=1)  # Concatenate features along the channel dimension
        x = self.classifier(x)
        return x

class CNN1dAttn(nn.Module):
    def __init__(self, num_heads=4, attention_dim=64):
        super(CNN1dAttn, self).__init__()
        self.encoder = make_cnn_layers()
        
        self.attention = nn.MultiheadAttention(embed_dim=attention_dim, num_heads=num_heads, batch_first=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(448, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        
        # Prepare for attention layer
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, seq_length, feature_dim)
        
        # Apply attention
        attn_output, _ = self.attention(x, x, x)
        
        # Flatten the attention output
        attn_output = attn_output.permute(0, 2, 1).contiguous()
        attn_output = attn_output.view(attn_output.size(0), -1)
        
        # Classifier
        x = self.classifier(attn_output)
        return x

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

# ResNet
class Bottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, downsample=False):
        super(Bottleneck, self).__init__()
        self.stride = 2 if downsample else 1

        self.layer = nn.Sequential(
            nn.LazyConv1d(mid_channels, kernel_size=1, stride=self.stride),
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            nn.LazyConv1d(mid_channels, kernel_size=3, padding=1),
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            nn.LazyConv1d(out_channels, kernel_size=1),
            nn.LazyBatchNorm1d(),
            nn.ReLU()
        )

        if in_channels != out_channels or downsample:
            self.res_layer = nn.LazyConv1d(out_channels, kernel_size=1, stride=self.stride)
            self.res_bn = nn.LazyBatchNorm1d()
        else:
            self.res_layer = None

    def forward(self, x):
        if self.res_layer is not None:
            residual = self.res_bn(self.res_layer(x))
        else:
            residual = x
        return self.layer(x) + residual


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.features = nn.Sequential(
            nn.LazyConv1d(64, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),

            Bottleneck(64, 64, 256, downsample=False),
            Bottleneck(256, 64, 256, downsample=False),
            Bottleneck(256, 64, 256, downsample=False),
            
            Bottleneck(256, 128, 512, downsample=True),
            Bottleneck(512, 128, 512, downsample=False),
            Bottleneck(512, 128, 512, downsample=False),
            Bottleneck(512, 128, 512, downsample=False),
            
            Bottleneck(512, 256, 1024, downsample=True),
            Bottleneck(1024, 256, 1024, downsample=False),
            Bottleneck(1024, 256, 1024, downsample=False),
            Bottleneck(1024, 256, 1024, downsample=False),
            Bottleneck(1024, 256, 1024, downsample=False),
            Bottleneck(1024, 256, 1024, downsample=False),
            
            Bottleneck(1024, 512, 2048, downsample=True),
            Bottleneck(2048, 512, 2048, downsample=False),
            Bottleneck(2048, 512, 2048, downsample=False),

            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)
        return x




class FusionModel(nn.Module):
    def __init__(self, model1, model2):
        super(FusionModel, self).__init__()

        # 각 모델의 마지막 Linear, Sigmoid 층 제거
        model1.classifier = nn.Sequential(*list(model1.classifier.children())[:-2])
        model2.classifier = nn.Sequential(*list(model2.classifier.children())[:-2])
        self.model1 = model1
        self.model2 = model2

        self.classifier = nn.Sequential(
            nn.LazyLinear(32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_other = x[:, :-182]
        x_voice = x[:, -182:]
        x_other = self.model1(x_other)
        x_voice = self.model2(x_voice)
        x = torch.cat((x_other, x_voice), dim=1)
        x = self.classifier(x)
        return x


class FusionBase(nn.Module):
    def __init__(self, models, input_slices):
        super(FusionBase, self).__init__()
        self.models = nn.ModuleList(models)
        self.input_slices = input_slices

        # Removing the last Linear and Sigmoid layers from each model
        for model in self.models:
            model.classifier = nn.Sequential(*list(model.classifier.children())[:-2])

        self.classifier = nn.Sequential(
            nn.LazyLinear(32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        outputs = []
        for model, slice_ in zip(self.models, self.input_slices):
            x_input = x[:, slice_]
            output = model(x_input)
            outputs.append(output)
        
        x = torch.cat(outputs, dim=1)
        x = self.classifier(x)
        return x

model1, model2, model3, model4 = DNN_small(), DNN_small(), DNN_small(), DNN_small()

fusion_other_voice = FusionBase(
    models=[model1, model2],
    input_slices=[slice(None, -182), slice(-182, None)]
)

fusion_s_w_iot_voice = FusionBase(
    models=[model1, model2, model3, model4],
    input_slices=[slice(None, -351), slice(-351, -342), slice(-342, -182), slice(-182, None)]
)
