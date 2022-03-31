# -*- coding: utf-8 -*-
"""
CASE STUDY CODE
The two models chosen: **DenseNet** and **VGG-16** 
Author: Pablo Garcia Fernández
"""


import matplotlib.pyplot as plt
import numpy as np
import os
import time
import copy

from torchinfo import summary
import torch
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""Training loop"""

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        init_epoch_time = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            #torch.save(model.state_dict(), os.path.join('./models', str(epoch) + '_model.pt'))
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
            
        print(f"Time per epoch: {time.time() - init_epoch_time}")
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def eval_model(model, testloader, criterion):
    since = time.time()
    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in testloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(testloader.dataset)
    epoch_acc = running_corrects.double() / len(testloader.dataset)
    print('{} Loss: {:.4f} Acc: {:.4f}'.format('test', epoch_loss, epoch_acc))
    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


# SHOW DATASET
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


"""### DenseNet-121 model"""

def densenet_121(train=False):

    # configuration
    # Number of classes in the dataset
    num_classes = 10

    # Batch size for training (change depending on how much memory you have)
    batch_size = 32

    # Number of epochs to train for
    num_epochs = 10

    # Same data
    # Transform to resize data to VGG dimensions
    transform=transforms.Compose([
                        transforms.Resize((224, 224)), 
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])

    # Download and load the training data
    trainset = datasets.FashionMNIST('./data', download=True, train=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # Download and load the test data
    validationset = datasets.FashionMNIST('./data', download=True, train=False, transform=transform)
    val_loader = DataLoader(validationset, batch_size=batch_size, shuffle=True)

    dataloaders_dict = {}
    dataloaders_dict['train'] = train_loader
    dataloaders_dict['val'] = val_loader

    model = models.densenet121(pretrained=False)

    # Change final linear layer to 10 outpus
    model.classifier = torch.nn.Linear(1024, num_classes)
    print(model)
    model = model.to(device)

    params_to_update = model.parameters()

    # Para comprobar que todos los parametros tienen el grad en True
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

    #optimizer = torch.optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(params_to_update, weight_decay=1e-4)

    # Setup the loss fxn
    criterion = torch.nn.CrossEntropyLoss()

    if train:
        # Train and evaluate
        model, hist = train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)
        # Save
        #torch.save(model.state_dict(), os.path.join('./models','final_model_densenet_10epoc.pt'))
        print(hist)

    else:
        summary(model, input_size=(batch_size, 3, 224, 224))
        model.load_state_dict(torch.load(os.path.join('./models','final_model_densenet_10epoc.pt')))        
        eval_model(model, val_loader, criterion)

### Change to True for training
densenet_121(False)