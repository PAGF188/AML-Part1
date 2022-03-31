# -*- coding: utf-8 -*-
"""
# CASE STUDY CODE
The two models chosen: **DenseNet** and **VGG-16** 
Author: Pablo Garcia Fernández
"""


import matplotlib.pyplot as plt
import numpy as np
import os
import time
import copy

import torch
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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


def MNIST(batch_size):
    # As before, we have to resize the images (224,224) and extend to 3 channels
    transform=transforms.Compose([
                        transforms.Resize((224, 224)), 
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])

    # Download and load the training data
    trainset = datasets.MNIST('./data', download=True, train=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # Download and load the test data
    validationset = datasets.MNIST('./data', download=True, train=False, transform=transform)
    val_loader = DataLoader(validationset, batch_size=batch_size, shuffle=True)

    dataloaders_dict = {}
    dataloaders_dict['train'] = train_loader
    dataloaders_dict['val'] = val_loader
    return dataloaders_dict



##################################################################################
# 1st bad approach -> 1st bad approach -> apply the mode directly on the new data.
# 1st bad approach -> 1st bad approach -> apply the mode directly on the new data.
def approach1():
    # configuration
    num_classes = 10
    batch_size = 32
    criterion = torch.nn.CrossEntropyLoss()

    # Data
    dataloaders_dict = MNIST(batch_size)

    # Model
    model = models.vgg16(pretrained=False)
    model.classifier[6] = torch.nn.Linear(4096, num_classes)
    print(model)
    model.load_state_dict(torch.load(os.path.join('./models','final_model_vgg_10epocs.pt')))        
    model = model.to(device)

    # Test
    eval_model(model, dataloaders_dict['val'], criterion)


##################################################################################
# 2ndgood approach -> train classifier (the fully connected layers)
# 2nd bad approach -> train classifier (the fully connected layers)
def approach2(train=False):
    # configuration
    num_classes = 10
    batch_size = 64
    criterion = torch.nn.CrossEntropyLoss()
    num_epochs = 10

    # Data
    dataloaders_dict = MNIST(batch_size)

    # Model
    model = models.vgg16(pretrained=False)
    model.classifier[6] = torch.nn.Linear(4096, num_classes)
    model.load_state_dict(torch.load(os.path.join('./models','final_model_vgg_10epocs.pt')))        
    
    # Freeze model
    for param in model.parameters():
        param.requires_grad = False

    # Add a new classifier
    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model.classifier[0].in_features
    model.classifier[0] = torch.nn.Linear(num_ftrs, 4096)
    model.classifier[3] = torch.nn.Linear(4096, 4096)
    model.classifier[6] = torch.nn.Linear(4096, num_classes)
    model = model.to(device)
    print(model)

    optimizer = torch.optim.SGD(model.classifier.parameters(), lr=0.01, momentum=0.9)

    if train:
        # Train and evaluate
        model, hist = train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)
        # Save
        torch.save(model.state_dict(), os.path.join('./models','f_vgg_tuned.pt'))
        print(hist)
    else:
        model.load_state_dict(torch.load(os.path.join('./models','f_vgg_tuned.pt')))        
        eval_model(model, dataloaders_dict['val'], criterion)


## Comentar para elegir approach
#approach1()
approach2(train=False)  # True para entrenar









