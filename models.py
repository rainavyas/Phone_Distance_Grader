import torch

class FCC(torch.nn.Module):
    def __init__(self, num_features):

        super(FCC, self).__init__()
        self.layer1 = torch.nn.Linear(num_features, 500)
        self.layer2 = torch.nn.Linear(500, 200)
        self.layer3 = torch.nn.Linear(200, 10)
        self.layer4 = torch.nn.Linear(10, 1)

    def forward(self, X):
        r = torch.nn.ReLU()
        h1 = r(self.layer1(X))
        h2 = r(self.layer2(h1))
        h3 = r(self.layer3(h2))
        y = self.layer4(h3)
        print(h1)
        return y.squeeze()
