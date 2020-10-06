import torch
import torch.nn.functional as F
import time
class FCC(torch.nn.Module):
    def __init__(self, num_features):

        super(FCC, self).__init__()
        self.layer1 = torch.nn.Linear(num_features, 500)
        self.layer2 = torch.nn.Linear(500, 200)
        self.layer3 = torch.nn.Linear(200, 10)
        self.layer4 = torch.nn.Linear(10, 1)

    def forward(self, X):

        # Check input is nan
        #nan_list = torch.isnan(torch.sum(X, 1)).tolist()
        #if True in nan_list:
        #    print("some values from input are nan")
        time.sleep(2)
        h1 = F.relu(self.layer1(X))
        print(h1[0][:10])
        h2 = F.relu(self.layer2(h1))
        h3 = F.relu(self.layer3(h2))
        y = self.layer4(h3)
        return y.squeeze()
