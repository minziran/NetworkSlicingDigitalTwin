from torch import nn

class LSTMModel(nn.Module):
    def __init__(self, input_features=14, hidden_features=64, output_features=14, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_features,
            batch_first=True,
            num_layers=num_layers
        )
        self.linear = nn.Linear(in_features=hidden_features, out_features=output_features)

    def forward(self, x):
        batch_size = x.shape[0]  # x is (batch_size, sequense_len, input_features)
        _, (hn, _) = self.lstm(x)
        out = self.linear(hn[-1])  # First dim of hn is num_layers

        return out


class LSTM_multitask(nn.Module):
    def __init__(self, input_features=14, hidden_features=128, output_category=10, output_noncategory=14, num_layers=1, dropout=0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_features,
            batch_first=True,
            num_layers=num_layers
        )
        self.dropout_p = dropout
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.linear_category = nn.Linear(in_features=hidden_features, out_features=output_category)
        self.linear_noncategory = nn.Linear(in_features=hidden_features, out_features=output_noncategory)


    def forward(self, x):
        batch_size = x.shape[0]  # x is (batch_size, sequense_len, input_features)
        _, (hn, _) = self.lstm(x)
        if self.dropout_p > 0:
            hn = self.dropout(hn)
        category_pred = self.linear_category(hn[-1])
        noncategory_pred = self.linear_noncategory(hn[-1])  # First dim of hn is num_layers
        return category_pred, noncategory_pred
