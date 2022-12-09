import argparse
import numpy as np
import torch as to
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from torch import nn, optim
from workspace_utils import active_session
from collections import OrderedDict
from PIL import Image

alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

models = {'alexnet': alexnet, 'vgg': vgg16}

def get_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type = str, help = 'Image Folder with default value "flowers"')
    parser.add_argument('--save_dir', type = str, default = '~/opt/', help = 'folder to save trained network')
    parser.add_argument('--arch', type = str, default = 'vgg', help = 'CNN Model Architecture')
    parser.add_argument('--learning_rate', type = float, default = 0.01, help = 'Learning rate of the network')
    parser.add_argument('--hidden_units', type = int, default = 512, help = 'Number of hidden layers to use for the training')
    parser.add_argument('--epochs', type = int, default = 10, help = 'Number of epoch to use for the training')
    parser.add_argument('--gpu', type = bool, default = True, help = 'Whether to use gpu or cpu')
    

    return parser.parse_args()
    
def model(arch, learning_rate, hidden_units):

    model = models[arch]

    for parameter in model.parameters():
        parameter.requires_grad = False

    model.classifier[1].requires_grad = True
    model.classifier[1] = to.nn.Linear(4096, hidden_units)
    model.classifier[3].requires_grad = True
    model.classifier[3] = to.nn.Linear(hidden_units, hidden_units)
    model.classifier[6] = to.nn.Linear(hidden_units, 102)

    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
    optimiser = optim.SGD(params_to_update, learning_rate)
    return model, optimiser

def save_checkpoint(model, path):
    to.save({'state_dict':model.state_dict(),
             'optimizer_state_dict':optimiser.state_dict(),
             'class_mapping': train_data.class_to_idx},
            path + '/my_checkpoint2.pth')

def train(data_dir, model, epochs, gpu): 
    epochs = epochs
    train_loss = 0
    step = 0
    
    with active_session():
    
        for e in range(epochs):

            model.train()
            train_loader, valid_loader = process(data_dir)
            
            for image, label in train_loader:
                step += 1
                image, label = image.to(device), label.to(device)
                model.to(device)

                ps = model(image)
                loss = nn.functional.cross_entropy(ps, label)

                optimiser.zero_grad()

                loss.backward()
                optimiser.step()

                train_loss += loss.item()

            model.eval()

            with to.no_grad():
                validation_loss = 0
                accuracy = 0
                correct_pred = 0

                for image, label in valid_loader:

                    image, label = image.to(device), label.to(device)

                    ps = model(image)
                    loss = nn.functional.cross_entropy(ps, label)
                    validation_loss += loss.item()

                    _, predicted_labels = to.max(ps, 1)

                    correct_pred += (predicted_labels == label).sum()
                    accuracy1 = correct_pred.float()/len(valid_loader) * 100

                    top_ps, top_class = ps.topk(1, dim=1)
                    equals = top_class == label.view(*top_class.shape)
                    accuracy += to.mean(equals.type(to.FloatTensor))

                print(f"Epochs: {e+1}/{epochs}..",
                      f"Training Loss: {train_loss/len(train_loader): .3f}..."
                      f"Validation Loss: {validation_loss/len(valid_loader): .3f}..."
                      f"Validation Accuracy: {accuracy/len(valid_loader): .3f}...")
                train_losss = 0
                model.train()

    return model

def process(data_dir):
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                std=[0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(data_dir, transform = train_transforms)
    valid_data = datasets.ImageFolder(data_dir, transform = valid_transforms)

    train_loader = to.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = to.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
    
    return train_loader, valid_loader

in_arg = get_input_args()

if in_arg.gpu:
    device = to.device('cuda:0')
else:
    #device = to.device("cuda:0" if to.cuda.is_available() else "cpu")
    device = to.device('cpu')

model, optimiser = model(in_arg.arch, in_arg.learning_rate, in_arg.hidden_units)

model = train(in_arg.data_dir, model, in_arg.epochs, in_arg.gpu)

save_checkpoint(model, in_arg.save_dir)
