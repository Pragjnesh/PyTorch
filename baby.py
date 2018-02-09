import torch, copy
import torch.nn as nn
import torch.optim as optim

from torchvision import models,transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from caltech256 import Caltech256


transform = transforms.Compose(
    [
        transforms.Scale((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)
data_dir = '/datasets/Caltech256/256_ObjectCategories/'
caltech256_train = Caltech256(data_dir, transform, train=True)
caltech256_test = Caltech256(data_dir, transform, train=False)

train_data = DataLoader(dataset = 'caltech256_train', batch_size = 32, shuffle = True, num_workers = 4)
test_data = DataLoader(dataset = 'caltech256_test', batch_size = 8, shuffle = True, num_workers = 4)

use_gpu = torch.cuda.is_available()

def train_model(model, criterion, optimizer, scheduler, num_epochs=5):
    model.train(True)
    for epoch in range(num_epochs):
        print(epoch)
        for data in train_data:
            scheduler.step()
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs,labels.view(-1).long())
            loss.backward()
            optimizer.step()

model_conv = models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 256)
model_conv = model_conv.cuda()
criterion = nn.CrossEntropyLoss()
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
model_conv = train_model(model_conv, criterion, optimizer_conv,exp_lr_scheduler, num_epochs=2)
