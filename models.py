import torch
import torch.nn.functional as F

class FCC(torch.nn.Module):
    def __init__(self, num_features):

        super(FCC, self).__init__()
        self.bn1 = torch.nn.BatchNorm1d(num_features=num_features)
        self.layer1 = torch.nn.Linear(num_features, 1000)
        self.layer2 = torch.nn.Linear(1000, 300)
        self.layer3 = torch.nn.Linear(300, 20)
        self.layer4 = torch.nn.Linear(20, 1)

    def forward(self, X):

        # Check if input has nan
        nan_list = torch.isnan(torch.sum(X, 1)).tolist()
        if True in nan_list:
           print("some values from input are nan")

        # Normalize each input vector
        X_norm = self.bn1(X)

        h1 = F.relu(self.layer1(X_norm))
        h2 = F.relu(self.layer2(h1))
        h3 = F.relu(self.layer3(h2))
        y = self.layer4(h3)
        return y.squeeze()
