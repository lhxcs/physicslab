import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)

# labels_map = {
#     0: "T-Shirt",
#     1: "Trouser",
#     2: "Pullover",
#     3: "Dress",
#     4: "Coat",
#     5: "Sandal",
#     6: "Shirt",
#     7: "Sneaker",
#     8: "Bag",
#     9: "Ankle Boot",
# }

# figure = plt.figure(figsize=(8,8))
# cols,rows = 3,3
# for i in range(1,cols*rows+1):
#     sample_idx = torch.randint(len(training_data),size=(1,)).item()
#     img, label = training_data[sample_idx]
#     figure.add_subplot(rows,cols,i)
#     plt.title(labels_map[label])
#     plt.axis("off")
#     plt.imshow(img.squeeze(),cmap='gray')
# plt.show()





# Simple Neural Network
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10),
        )

    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork().to(device)

learning_rate = 1e-3
batch_size = 64
epochs = 15

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X,y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred,y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 ==0:
            loss, current = loss.item(),(batch+1)*len(X)
            print(f"loss:{loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader,model,loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X,y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1)==y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy:{(100*correct):>0.1f}%,Avg loss: {test_loss:>8f} \n ")


for t in range(epochs):
    print(f"Epoch  {t+1}\n-------------------------------")
    train_loop(train_dataloader,model,loss_fn,optimizer)
    test_loop(test_dataloader,model,loss_fn)
print("Done!")
# X = torch.rand(1,28,28,device=device)()
# logits = model(X)
# pred_probab = nn.Softmax(dim=1)(logits)
# y_pred = pred_probab.argmax(1)
# # print(f"Predicted class:{y_pred}")

# input_image = torch.rand(3,28,28)
# flatten = nn.Flatten()
# flat_image = flatten(input_image)
# # print(flat_image.size())
# layer1 = nn.Linear(in_features=28*28, out_features=20)
# hidden1 = layer1(flat_image)
# # print(hidden1.size())
# # print(f"Before ReLU:{hidden1}\n\n")
# hidden1 = nn.ReLU()(hidden1)
# # print(f"After ReLU:{hidden1}")

# seq_modules = nn.Sequential(
#     flatten,
#     layer1,
#     nn.ReLU(),
#     nn.Linear(20, 10)
# )
# input_image = torch.rand(3,28,28)
# logits = seq_modules(input_image)