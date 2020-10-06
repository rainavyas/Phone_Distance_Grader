import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from models import FCC
import pickle


input_file = "BLXXXgrd02.pkl"
pkl = pickle.load(open(input_file, "rb"))

# Construct feature vectors tensor
X = []

for spk in range(len(pkl['plp'])):
    X.append(pkl['pdf'][spk])

# Construct the output scores tensor
print(pkl.keys())
