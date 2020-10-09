import torch
import pickle
from utility import *


model_path = 'FCC_lpron_1.pt'
input_file = "BLXXXeval3.pkl"
pkl = pickle.load(open(input_file, "rb"))

# Construct feature vectors tensor
X = []

for spk in range(len(pkl['plp'])):
    feats = pkl['pdf'][spk]
    log_feats = [math.log(feat) if feat != -1 else feat for feat in feats]
    # log all the features
    X.append(log_feats)

# Construct the output scores tensor
y_list = (pkl['score'])
y = torch.FloatTensor(y_list)

X = torch.FloatTensor(X)

# Load the model
model = torch.load(model_path)
model.eval()

y_pred = model(X)
y_pred[y_pred>6]=6.0
y_pred[y_pred<0]=0.0
y_pred_list = y_pred.tolist()

mse = calculate_mse(y_pred_list, y_list)
pcc = calculate_pcc(y_pred, y)
less1 = calculate_less1(y_pred, y)
less05 = calculate_less05(y_pred, y)

print("mse: "+ str(mse)+"\n pcc: "+str(pcc)+"\n less than 1 away: "+ str(less1)+"\n less than 0.5 away: "+str(less05))
