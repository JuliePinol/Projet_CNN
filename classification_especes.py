from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from PIL import Image
from torch.autograd import Variable
import split_folders

####### Code copié et modifié a partir du code de Tsaku Nelson : 
#https://medium.com/@tsakunelsonz/loading-and-training-a-neural-network-with-custom-dataset-via-transfer-learning-in-pytorch-8e672933469

#Recuperer un modele deja existant (vgg16 ici)
model = models.vgg16(pretrained = True)


#creer une couche de classification
classifier =nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 512)),### couche d'input. Ici 25088 = input size du NN
                           ('relu', nn.ReLU()), ## Type de fonction sigma utilisee
                           ('dropout', nn.Dropout(p=0.337)), # Taux de dropout des neurones
                           ('fc2', nn.Linear(512, 4)),#### Couche de sortie. Ici 4 = output size du NN
                           ('output', nn.LogSoftmax(dim=1))
                             ]))

#remplacer le classificateur du modele avec celui-ci : (transfer learning)
model.classifier = classifier


#definir les criteres d'optimisation
criteria = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.005, momentum = 0.5)

#define training function
def train (model, loader, criterion, gpu):
    model.train()
    current_loss = 0 #Sommes de loss sur chaque iteration
    current_correct = 0
    for train, y_train in iter(loader):
        if gpu: #Effectuer le calcul sur la carte graphique
            train, y_train = train.to('cuda'), y_train.to('cuda')
        optimizer.zero_grad() #Reinitialiser le gradient pour l'optimisation
        output = model.forward(train) #phase de forward pass
        _, preds = torch.max(output,1)
        loss = criterion(output, y_train) # calcul de la loss
        loss.backward()#backward pass
        optimizer.step() #modification des poids
        current_loss += loss.item()*train.size(0) #actualisation des loss sommees
        current_correct += torch.sum(preds == y_train.data)
    epoch_loss = current_loss / len(dataloaders["train"].dataset) #actualisation du message
    epoch_acc = current_correct.double() / len(dataloaders["train"].dataset)
        
    return epoch_loss, epoch_acc

#define validation function
def validation (model, loader, criterion, gpu):
    model.eval()
    valid_loss = 0 #Idem, initialiser les loss
    valid_correct = 0
    for valid, y_valid in iter(loader):
        if gpu:
            valid, y_valid = valid.to('cuda'), y_valid.to('cuda')
            
        output = model.forward(valid) # Forward pass sur les donnees de validation
        valid_loss += criterion(output, y_valid).item()*valid.size(0) # Calcul de la loss
        equal = (output.max(dim=1)[1] == y_valid.data) # Comparaison entre loss de validation et output du modele
        valid_correct += torch.sum(equal)#type(torch.FloatTensor)
    
    epoch_loss = valid_loss / len(dataloaders["val"].dataset) #Actualisation du message
    epoch_acc = valid_correct.double() / len(dataloaders["val"].dataset)
    
    return epoch_loss, epoch_acc


################## Charger les donnees

#Création du split du dataset
split_folders.ratio('/Users/juliepinol/Desktop/dataset', output='/Users/juliepinol/Desktop/dataset_split', seed=1337, ratio=(.8, .1, .1)) # default values 
 
data_dir = "/Users/juliepinol/Desktop/dataset_split" #repertoire du dataset
#Le dataset doit etre organise selon un format de dossiers dataset > train/test/valid > classes

data_transforms = { #On definit les transformations pour l'input de ImageFolder 
    'train': transforms.Compose([ #On definit les transfo des images
        transforms.RandomResizedCrop(224), #cropping aleatoire des bords de l'image
        transforms.RandomHorizontalFlip(), #flips aleatoires de l'image
        transforms.ToTensor(), # transformation en tensor, obligatoire pour pytorch
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #normalisation de l'input pour vgg16
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

#recupererer les objets contenus dans le repertoire et les transformer   
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

#definir un objet loader permettant au NN d'importer des batchs pour l'entrainement
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}

#Definir les dimensions et classes du dataset
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

#Empecher les changements de gradients dans les parametres du modele
for param in model.parameters():
    param.require_grad = False
    
#send model to GPU
if torch.cuda.is_available():
    model.to('cuda')
    
#definir le coeur de calcul
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#######Apprentissage et validation :
epochs = 10  
epoch = 0


    
#Boucle d'apprentissage : Pour chaque epoque on appelle les fonctions d'entrainement et de validation
for e in range(epochs):
    epoch +=1
    print(epoch)
    with torch.set_grad_enabled(True):
        epoch_train_loss, epoch_train_acc = train(model,dataloaders["train"], criteria, torch.cuda.is_available())
        print("Epoch: {} Train Loss : {:.4f}  Train Accuracy: {:.4f}".format(epoch,epoch_train_loss,epoch_train_acc))
    with torch.no_grad():
        epoch_val_loss, epoch_val_acc = validation(model, dataloaders["val"], criteria, torch.cuda.is_available())
        print("Epoch: {} Validation Loss : {:.4f}  Validation Accuracy {:.4f}".format(epoch,epoch_val_loss,epoch_val_acc))



## Evaluation du modele
model.eval()
total = 0
correct = 0 
count = 0
#iterating for each sample in the test dataset once
for test, y_test in iter(dataloaders["train"]):
    if torch.cuda.is_available():
        test, y_test = test.to('cuda'), y_test.to('cuda')
#Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        output = model.forward(test)
        ps = torch.exp(output)
        _, predicted = torch.max(output.data,1)
        total += y_test.size(0)
        correct += (predicted == y_test).sum().item() 
        count += 1
        print("Accuracy of network on test images is ... {:.4f}....count: {}".format(100*correct/total,  count ))


#### Enregistrer les poids du modèle :
torch.save(model.state_dict(), "/Users/juliepinol/Desktop/vgg16_pretrained")


### Recharger le modele :
model.load_state_dict(torch.load("/Users/juliepinol/Desktop/vgg16_pretrained"))


##### Test du modele :      


loader = transforms.Compose([transforms.Resize(256),
                             transforms.ToTensor(), # transformation en tensor, obligatoire pour pytorch
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

### Code recupere : https://discuss.pytorch.org/t/how-to-classify-single-image-using-loaded-net/1411/3
def image_loader(image, loader):
    """load image, returns cuda tensor"""
    image = loader(image).float()
    image = Variable(image, requires_grad=False) #ici on veut seulement classifier les images, pas modifier les gradients
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image  


### Test du modele sur une image
image=Image.open("/Users/juliepinol/Desktop/dataset_split/test/corn/19MAIS0000016_00002.jpg")
image = image_loader(image, loader)
predicted=model(image)
predicted=predicted.max(1, keepdim=True)[1]
label=class_names[predicted]
print(label)



