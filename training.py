import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from models import FCC, FCC_shallow
import pickle
from utility import calculate_mse


# Define constants
lr = 3*1e-2
epochs = 800
bs = 450
seed = 1

torch.manual_seed(seed)

input_file = "BLXXXgrd02.pkl"
output_file = "FCC_pron_"+str(seed)+".pt"
pkl = pickle.load(open(input_file, "rb"))

# Construct feature vectors tensor
X = []

for spk in range(len(pkl['plp'])):
    X.append(pkl['pdf'][spk])

# Construct the output scores tensor
y = (pkl['score'])
num_features = len(X[0])


# Split into training and dev sets
num_dev = 100
X_dev = X[:num_dev]
X_train = X[num_dev:]
y_dev = y[:num_dev]
y_train = y[num_dev:]

X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_dev = torch.FloatTensor(X_dev)
y_dev = torch.FloatTensor(y_dev)


# Store all training dataset in a single wrapped tensor
train_ds = TensorDataset(X_train, y_train)

# Use DataLoader to handle minibatches easily
train_dl = DataLoader(train_ds, batch_size = bs, shuffle = True)

model = FCC(num_features)
print("model initialised")

criterion = torch.nn.MSELoss(reduction = 'mean')
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
#optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# Define a scheduler for an adaptive learning rate
lambda1 = lambda epoch: 0.99**epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda1)

for epoch in range(epochs):
    model.train()
    for xb, yb in train_dl:

        # Forward pass
        y_pred = model(xb)

        # Compute loss
        loss = criterion(y_pred, yb)

        # Zero gradients, backward pass, update weights
        optimizer.zero_grad()
        loss.backward()
        #optimizer.step()
        scheduler.step()

    print(loss.item())  
    model.eval()
    # Evaluate on dev set
    y_pr = model(X_dev)
    y_pr[y_pr>6]=6
    y_pr[y_pr<0]=0
    dev_loss = calculate_mse(y_pr.tolist(), y_dev.tolist())
    print(epoch, dev_loss)


# Save the model to a file
torch.save(model, output_file)
