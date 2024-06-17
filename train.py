import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from network import SuccessiveConvolutions
from dataset import SparseViewSinograms

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"  # this is for mac (Metal Performance Shaders)
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

model = SuccessiveConvolutions(n_layers=20)

learning_rate = 1e-4
batch_size = 67
epochs = 200

# Note: in original paper loss is divided by 2
loss_fn = torch.nn.MSELoss()
# loss_fn = 0.5 * torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

training_data = SparseViewSinograms(
    input_path=Path("/lhome/clarkcs/aRTist_simulations/aRTist_train_data/sinograms"),
    filename="train_full_sinogram"
)
train_dataloader = DataLoader(training_data, batch_size=batch_size)

test_data = SparseViewSinograms(
    input_path=Path("/lhome/clarkcs/aRTist_simulations/aRTist_test_data/sinograms"),
    filename="test_full_sinogram"
)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    # set to training mode
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(x)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # set to evaluation mode
    model.eval()
    # size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
         for x, y in dataloader:
              pred = model(x)
              test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    return test_loss


# train_losses = []
test_losses = []
for e in range(epochs):
    print(f"Epoch {e+1} ----------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    if e % 20 == 0:
        test_loss = test_loop(test_dataloader, model, loss_fn)
        test_losses.append(test_loss)

plt.plot(list(range(0, epochs, 20)), test_losses, "r*")
plt.savefig("losses.png")

torch.save(model.state_dict(), "model_weights.pth")
