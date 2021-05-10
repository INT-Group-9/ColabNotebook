import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import tqdm
import visualdl
import torch.onnx

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

trainSet = datasets.CIFAR10("./data", train=True, download = False,
                         transform = transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomCrop(32, padding=4),
                            transforms.ToTensor()]))

testSet = datasets.CIFAR10("./data", train=False, download=False,
                         transform = transforms.Compose([transforms.ToTensor()]))

trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=64, shuffle=True)
testLoader = torch.utils.data.DataLoader(testSet, batch_size=64, shuffle=True)

class Net(nn.Module):

  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
    self.batch1 = nn.BatchNorm2d(32)

    self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
    self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
    self.batch2 = nn.BatchNorm2d(128)

    self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
    self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
    self.batch3 = nn.BatchNorm2d(256)


    self.fc1 = nn.Linear(in_features = 256 * 4 * 4, out_features=1024)
    self.fc2 = nn.Linear(in_features = 1024, out_features=512)
    self.fc3 = nn.Linear(in_features = 512, out_features=10)
    self.dropout1 = nn.Dropout(p=0.1)
    self.dropout2 = nn.Dropout(p=0.05)

    self.pool = nn.MaxPool2d(2, 2)

  def forward(self, input):
    output = F.relu(self.conv1(input))
    output = self.batch1(output)
    # output = F.relu(output)
    output = F.relu(self.conv2(output))
    output = self.pool(output)

    output = F.relu(self.conv3(output))
    output = self.batch2(output)
    # output = F.relu(output)
    output = F.relu(self.conv4(output))
    output = self.pool(output)
    output = self.dropout2(output)

    output = F.relu(self.conv5(output))
    output = self.batch3(output)
    # output = F.relu(output)
    output = F.relu(self.conv6(output))
    output = self.pool(output)

    output = output.view(output.size(0), -1)

    output = self.dropout1(output)
    output = F.relu(self.fc1(output))
    output = F.relu(self.fc2(output))
    output = self.dropout1(output)
    output = self.fc3(output)
    return output
    
net = Net().cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

writer = SummaryWriter()
for epoch in range(1):  # loop over the dataset multiple times
    print("epoch", epoch + 1)
    running_loss = 0.0
    for i, data in enumerate(trainLoader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        writer.add_scalar("Loss/train", loss, epoch)
        loss.backward()
        optimizer.step()

print('Finished Training')
writer.flush()

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

dataiter = iter(testLoader)
images, labels = dataiter.next()
images = images.cuda()
labels = labels.cuda()

print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

net = Net().cuda()
net.load_state_dict(torch.load(PATH))
outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))


correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testLoader:
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


#DIAGRAM

batch = next(iter(trainLoader))
inputs, labels = batch
inputs, labels = inputs.cuda(), labels.cuda()

# writer.add_graph(net, inputs)
# writer.close()
torch.onnx.export(net,               # model being run
                  inputs,                         # model input (or a tuple for multiple inputs)
                  "super_resolution.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

