from torch import nn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import torch

# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

# Create circles
X, y = make_circles(n_samples = 1000,
                    noise=0.03, # a little bit of noise to the dots
                    random_state=42) # keep random state so we get the same values

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# Split data into train and test sets


X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, # 20% test, 80% train
                                                    random_state=42) # make the random split reproducible

model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
)


loss_fn = nn.BCEWithLogitsLoss() # BCEWithLogitsLoss = sigmoid built-in
optimizer = torch.optim.SGD(params=model_0.parameters(), 
                            lr=0.1)

# View the frist 5 outputs of the forward pass on the test data
y_logits = model_0(X_test)[:5]

# Use sigmoid on model logits
y_pred_probs = torch.sigmoid(y_logits)

# Find the predicted labels (round the prediction probabilities)
y_preds = torch.round(y_pred_probs)
print ( f"\n{y_preds = }" ) , print ( f"{y_preds.shape = }" )

# In full
y_pred_labels = torch.round(torch.sigmoid(model_0(X_test)[:5]))

# Check for equality
print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))

# Get rid of extra dimension
y_preds.squeeze()
print ( f"\n{y_preds = }" ) , print ( f"{y_preds.shape = }" )