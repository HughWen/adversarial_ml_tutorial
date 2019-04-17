# %% Chapter 2 Linear models
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-4, 4)
print(x)
plt.plot(x, np.log(1 + np.exp(-x)))

# %% load MNIST dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

mnist_train = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())

train_idx = mnist_train.targets <= 1
mnist_train.data = mnist_train.data[train_idx]
mnist_train.targets = mnist_train.targets[train_idx]

test_idx = mnist_test.targets <= 1
mnist_test.data = mnist_test.data[test_idx]
mnist_test.targets = mnist_test.targets[test_idx]

train_loader = DataLoader(mnist_train, batch_size=100, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=100, shuffle=False)

# %%
import torch
import torch.nn as nn
import torch.optim as optim


def epoch(loader, model, opt=None):
    total_loss, total_err = 0.0, 0.0
    for X, y in loader:
        yp = model(X.view(X.shape[0], -1))[:, 0]
        loss = nn.BCEWithLogitsLoss()(yp, y.float())
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total_err += ((yp > 0) * (y == 0) + (yp < 0) * (y == 1)).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)


model = nn.Linear(784, 1)
opt = optim.SGD(model.parameters(), lr=1.0)
print("Train Err", "Train Loss", "Test Err", "Test Loss", sep="\t")
for i in range(10):
    train_err, train_loss = epoch(train_loader, model, opt)
    test_err, test_loss = epoch(test_loader, model)
    print(*("{:.6f}".format(i) for i in (train_err, train_loss, test_err, test_loss)), sep="\t")


# %%
X_test = (test_loader.dataset.test_data.float() / 255).view(len(test_loader.dataset), -1)
y_test = test_loader.dataset.test_labels
yp = model(X_test)[:, 0]
idx = (yp > 0) * (y_test == 0) + (yp < 0) * (y_test == 1)
plt.imshow(1 - X_test[idx][0].view(28, 28).numpy(), cmap='gray')
plt.title('True Label: {}'.format(y_test[idx].item()))

# %%
epsilon = 0.2
delta = epsilon * model.weight.detach().sign().view(28,28)
plt.imshow(1 - delta.numpy(), cmap='gray') # I don't know why here should use 1 - delta

# %% adversarial attack
def epoch_adv(loader, model, dleta):
    total_loss, total_err = 0., 0.
    for X, y in loader:
        yp = model((X - (2 * y.float()[:, None, None, None] - 1) * delta).view(X.shape[0], -1))[:, 0]
        loss = nn.BCEWithLogitsLoss()(yp, y.float())
        total_err += ((yp > 0) * (y == 0) + (yp < 0) * (y == 1)).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)
print(epoch_adv(test_loader, model, delta[None, None, :, :]))

# %% show the perturbed images
f,ax = plt.subplots(5,5, sharey=True)
for i in range(25):
    ax[i%5][i//5].imshow(1-(X_test[i].view(28,28) - (2*y_test[i]-1)*delta).numpy(), cmap="gray")


# %% Training robust linear models
def epoch_robust(loader, model, epsilon, opt=None):
    total_loss, total_err = 0.0, 0.0
    for X, y in loader:
        yp = model(X.view(X.shape[0], -1))[:, 0] - epsilon * (2 * y.float() - 1) * model.weight.norm(1)
        loss = nn.BCEWithLogitsLoss()(yp, y.float())
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total_err += ((yp > 0) * (y == 0) + (yp < 0) * (y == 1)).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)

model  = nn.Linear(784, 1)
opt = optim.SGD(model.parameters(), lr=1e-1)
epsilon = 0.2
print('Rob. Train Err', 'Rob. Train Loss', 'Rob. Test Err', 'Rob. Test Loss', sep='\t')
for i in range(30):
    train_err, train_loss = epoch_robust(train_loader, model, epsilon, opt)
    test_err, test_loss = epoch_robust(test_loader, model, epsilon)
    print(*('{:.6f}'.format(i) for i in (train_err, train_loss, test_err, test_loss)), sep='\t')


# %%
train_err, train_loss = epoch(train_loader, model)
test_err, test_loss = epoch(test_loader, model)
print("Train Err", "Train Loss", "Test Err", "Test Loss", sep="\t")
print(*("{:.6f}".format(i) for i in (train_err, train_loss, test_err, test_loss)), sep="\t")

# %% adversarial attack
print(epoch_adv(test_loader, model, delta[None, None, :, :]))

# %% show the final optimal perturbation
delta = epsilon * model.weight.detach().sign().view(28, 28)
plt.imshow(1 - delta.numpy(), cmap='gray')