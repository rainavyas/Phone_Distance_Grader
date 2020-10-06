import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from models import FCC
import pickle


input_file = "BLXXXgrd02.pkl"
pkl = pickle.load(open(input_file, "rb"))
