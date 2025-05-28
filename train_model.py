# train_model.py
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.layer(x)

model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('.', download=True, train=True, transform=transforms.ToTensor()),
    batch_size=64, shuffle=True)

for epoch in range(1):
    for imgs, labels in train_loader:
        output = model(imgs)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), 'app/model.pt')
