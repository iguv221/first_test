from __future__ import print_function
from datetime import datetime

import argparse
import os
import random

import random
import math
from random import randint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils import data
import torchvision.utils as vutils
from torch.utils.data import DataLoader as dataloader
from torch.utils.data import ConcatDataset

from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import warnings
import time
import copy
import traceback
from datetime import datetime
import umap.umap_ as umap

global ROUTE
ROUTE = __file__[:-7]

manualSeed=999
torch.manual_seed(manualSeed)

torch.set_printoptions(profile="full")
np.set_printoptions(threshold=10000)

GPU_NUM = 0
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available else 'cpu')
torch.cuda.set_device(device)

print(torch.cuda.get_device_name(0))
print(torch.cuda.is_available())
print(torch.__version__)

def show_loaded_images(loaded_data):
    real_batch = next(iter(loaded_data))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, para):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d( para.nz, para.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(para.ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(para.ngf * 8, para.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(para.ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d( para.ngf * 4, para.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(para.ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d( para.ngf * 2, para.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(para.ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d( para.ngf, para.nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
    
class Discriminator(nn.Module):
    def __init__(self, para):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(para.nc, para.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(para.ndf, para.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(para.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(para.ndf * 2, para.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(para.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(para.ndf * 4, para.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(para.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(para.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

class Run_DCGAN:
    def __init__(self,data,para,device):
        self.data = data
        self.para = para
        netG = Generator(self.para).to(device)
        netG.apply(weights_init)
        self.netG = netG
        netD = Discriminator(self.para).to(device)
        netD.apply(weights_init)
        self.netD = netD
        self.device = device

    def train(self):
        netD = self.netD
        netG = self.netG
        device = self.device
        criterion = nn.BCELoss()
        fixed_noise = torch.randn(64, self.para.nz, 1, 1, device=self.device)
        real_label = 1.
        fake_label = 0.
        optimizerD = optim.Adam(netD.parameters(), lr=self.para.lr, betas=(self.para.beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=self.para.lr, betas=(self.para.beta1, 0.999))
        G_losses = []
        D_losses = []
        iters = 0

        print("Starting Training Loop...")
        for epoch in range(self.para.num_epochs):
            for i, data in enumerate(self.data, 0):
                netD.zero_grad()
                real_cpu = data[0].to(device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                output = netD(real_cpu).view(-1)
                errD_real = criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()
                noise = torch.randn(b_size, self.para.nz, 1, 1, device=device)
                fake = netG(noise)
                label.fill_(fake_label)
                output = netD(fake.detach()).view(-1)
                errD_fake = criterion(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                optimizerD.step()
                netG.zero_grad()
                label.fill_(real_label) 
                output = netD(fake).view(-1)
                errG = criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                optimizerG.step()
                G_losses.append(errG.item())
                if errD.item()>99:
                    self.para.SUCCESS=False
                    return
                D_losses.append(errD.item())
            if (epoch+1)%max(1,(self.para.num_epochs)//3)==0:
                UMAP_Data = call_data()
                reduce(UMAP_Data,10000)
                UMAP_new_data = Just_Add(self.netG,UMAP_Data,500,11)
                Run_UMAP(UMAP_new_data,self.para.normal_class,epoch+1)
        self.G_losses = G_losses
        self.D_losses = D_losses
        self.netD = netD
        self.netG = netG

    def plot_loss(self):
        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.G_losses,label="G")
        plt.plot(self.D_losses,label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def show_images(self,test_num,save=False,route=""):
        for i, data in enumerate(self.data):
            real_cpu = data[0].to(device)
            break
        b_size = real_cpu.size(0)
        noise = torch.randn(b_size, self.para.nz, 1, 1, device=device)
        fake = self.netG(noise)
        img = vutils.make_grid(fake, padding=2, normalize=True).cpu()

        plt.figure(figsize=(15,15))
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(img,(1,2,0)))

        if save==True:
            now = datetime.now()
            time = now.strftime("%d-%m-%Y %H %M %S")
            Title =route+"  {},   Test Num {} , LR {} , Epochs {}.png".format(time,test_num,self.para.lr,self.para.num_epochs)
            plt.savefig(Title)
        plt.show()

    def return_generator(self):
        return self.netG



class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Sequential(
            nn.Linear(32*16*16,100),
            nn.Linear(100,10),
            nn.Linear(10,2),
        )
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        output = self.out(x)
        return output,x

class Classify_with_CNN:
    def __init__(self,args,data,device):
        self.args=args
        train,test = data
        self.train_data,self.test_data = train,test
        self.device = device
        
    def CNN_train(self):
        train_loader = self.train_data
        cnn = CNN().to(device)
        loss_func = nn.CrossEntropyLoss().to(device) 
        optimizer = optim.Adam(cnn.parameters(),lr=self.args.learning_rate)
        cnn.train()
        for epoch in range(self.args.num_epochs):
            for idx, (x,y) in enumerate(train_loader):
                x = x.to(device)
                y = y.type(torch.LongTensor).to(device)
                output = cnn(x)[0]
                loss = loss_func(output,y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.cnn = cnn
    
    def CNN_test(self):
        self.cnn.eval()
        test_loader=self.test_data
        with torch.no_grad():
            for images, labels in test_loader:
                images=images.to(device)
                labels=labels.to(device)
                test_output , last_layer =self.cnn(images)
                pred_y = torch.max(test_output,1)[1].data.squeeze()
                accuracy = (pred_y==labels).sum().item()/float(labels.size(0))
    
        self.accuracy=accuracy

    def get_scores(self):
        return self.accuracy


def weights_init_normal(m): 
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != 'Conv':
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

class MNIST_Dataset(data.Dataset):
    
    def __init__(self, data, targets, transform):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        if self.transform:
            x = Image.fromarray(x.detach().cpu().numpy())
            x = self.transform(x)
        return x, y

class pretrain_autoencoder(nn.Module):
    def __init__(self, z_dim=32):
        super(pretrain_autoencoder, self).__init__()
        self.z_dim = z_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4 * 7 * 7, z_dim, bias=False)

        self.deconv1 = nn.ConvTranspose2d(2, 4, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(4, 8, 5, bias=False, padding=3)
        self.bn4 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(8, 1, 5, bias=False, padding=2)
        
    def encoder(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        return self.fc1(x)
   
    def decoder(self, x):
        x = x.view(x.size(0), int(self.z_dim / 16), 4, 4)
        x = F.interpolate(F.leaky_relu(x), scale_factor=2)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=2)
        x = self.deconv3(x)
        return torch.sigmoid(x)
        
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

class network(nn.Module):
    def __init__(self, z_dim=32):
        super(network, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(1024,  10*z_dim, bias=False)
        self.fc2 = nn.Linear(10*z_dim,  z_dim, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return self.fc2(x)

def Fashion_Mnist_Transformer(small_class):
    transform = transforms.Compose(
        [transforms.ToTensor(),transforms.Normalize((0.5,),(0.5))]  
    ) 
    train = datasets.FashionMNIST(root='dataset/', train=True, download=True)
    test = datasets.FashionMNIST(root='dataset/', train=False, download=True)
    x_train = train.data
    y_train = train.targets
    x_train = x_train[np.where(y_train==small_class)]
    y_train = y_train[np.where(y_train==small_class)]                       
    data_train = MNIST_Dataset(x_train, y_train, transform)
    x_test = test.data
    y_test = test.targets
    y_test = np.where(y_test==small_class, 0, 1)
    data_test = MNIST_Dataset(x_test, y_test, transform)
    return data_train, data_test

class Deep_SVDD(nn.Module):
    def __init__(self, z_dim=32): 
        super(Deep_SVDD, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(1024,  10*z_dim, bias=False)
        self.fc2 = nn.Linear(10*z_dim,  z_dim, bias=False)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return self.fc2(x)


class C_AutoEncoder(nn.Module):
    def __init__(self, z_dim=32):
        super(C_AutoEncoder, self).__init__()
        self.z_dim = z_dim
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(1024,  10*z_dim, bias=False)
        self.fc2 = nn.Linear(10*z_dim,  z_dim, bias=False)
        self.deconv1 = nn.ConvTranspose2d(2, 4, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(4, 8, 5, bias=False, padding=2)
        self.bn4 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(8, 4, 5, bias=False, padding=2)
        self.bn5 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(4, 1, 5, bias=False, padding=2)


        
    def encoder(self, x):
        # print(x.size())
        # asda
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return self.fc2(x)
   
    def decoder(self, x):
        x = x.view(x.size(0), int(self.z_dim / 16), 4, 4)
        x = F.interpolate(F.leaky_relu(x), scale_factor=2)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bn5(x)), scale_factor=2)
        x = self.deconv4(x)
        return torch.sigmoid(x)
        
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

class TrainerDeepSVDD:
    def __init__(self, args, data, device):
        self.args = args
        self.train_loader, self.test_loader = data
        self.device = device
    
    def pretrain(self):
        ae = C_AutoEncoder(self.args.latent_dim).to(self.device)
        ae.apply(weights_init_normal)
        optimizer = optim.Adam(ae.parameters(), lr=self.args.lr_ae,
                               weight_decay=self.args.weight_decay_ae)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                    milestones=self.args.lr_milestones, gamma=0.1)

        ae.train()
        for epoch in range(self.args.num_epochs_ae):
            total_loss = 0
            for x, _ in (self.train_loader):
                x = x.float().to(self.device)
                optimizer.zero_grad()
                x_hat = ae(x)
                reconst_loss = torch.mean(torch.sum((x_hat - x) ** 2, dim=tuple(range(1, x_hat.dim()))))
                reconst_loss.backward()
                optimizer.step()
                total_loss += reconst_loss.item()
            scheduler.step()
        self.save_weights_for_DeepSVDD(ae, self.train_loader) 
        
    

    def save_weights_for_DeepSVDD(self, model, dataloader):
        c = self.set_c(model, dataloader)
        net = network(self.args.latent_dim).to(self.device)
        state_dict = model.state_dict()
        net.load_state_dict(state_dict, strict=False)
        torch.save({'center': c.cpu().data.numpy().tolist(),
                    'net_dict': net.state_dict()}, ROUTE+'pretrained_parameters.pth')
    
    def set_c(self, model, dataloader, eps=0.1):
        model.eval()
        z_ = []
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.float().to(self.device)
                z = model.encoder(x)
                z_.append(z.detach())
        z_ = torch.cat(z_)
        c = torch.mean(z_, dim=0)
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c


    def train(self):
        net = Deep_SVDD().to(self.device)
        if self.args.pretrain==True:
            state_dict = torch.load(ROUTE+'pretrained_parameters.pth')
            net.load_state_dict(state_dict['net_dict'])
            c = torch.Tensor(state_dict['center']).to(self.device)
        else:
            net.apply(weights_init_normal)
            c = torch.randn(self.args.latent_dim).to(self.device)
        
        optimizer = optim.Adam(net.parameters(), lr=self.args.lr,
                               weight_decay=self.args.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                    milestones=self.args.lr_milestones, gamma=0.1)

        net.train()
        for epoch in range(self.args.num_epochs):
            total_loss = 0
            for x, _ in (self.train_loader):
                x = x.float().to(self.device)
                optimizer.zero_grad()
                z = net(x)
                loss = torch.mean(torch.sum((z - c) ** 2, dim=1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
        self.net = net
        self.c = c

def eval(net, c, dataloader, device):
    scores = []
    labels = []
    net.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.float().to(device)
            z = net(x)
            score = torch.sum((z - c) ** 2, dim=1)
            scores.append(score.detach().cpu())
            labels.append(y.cpu())
    labels, scores = torch.cat(labels).numpy(), torch.cat(scores).numpy()
    roc_auc_scores = roc_auc_score(labels, scores)*100
    print('ROC AUC score: {:.2f}'.format(roc_auc_scores))
    return roc_auc_scores

def RUN_DEEPSVDD(Data, Args,device):
    print("\nRunning Deep SVDD with {} as anomaly value.".format(Args.normal_class))
    deep_SVDD = TrainerDeepSVDD(Args, Data, device)
    if Args.pretrain:
        deep_SVDD.pretrain()
    deep_SVDD.train()
    ROC_AUC_Scores = eval(deep_SVDD.net, deep_SVDD.c, Data[1], device)
    return ROC_AUC_Scores


def Run_UMAP(Data,anomaly_value,epoch_num):
    list_targets = []
    for tg in Data.targets:
        if tg == 11:
            list_targets.append(0)
        elif tg == anomaly_value:
            list_targets.append(1)
        else:
            list_targets.append(2)
    array_targets = np.array(list_targets)

    X_data = Data.data.reshape(len(Data),64*64)
    reducer = umap.UMAP(random_state=42) 
    u = reducer.fit(X_data.detach().cpu().numpy().squeeze())

    embedding = reducer.transform(X_data.detach().cpu().numpy().squeeze())
    assert(np.all(embedding == reducer.embedding_))
    plt.figure(figsize=(10,6))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=array_targets, cmap='Spectral',s=3)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
    plt.title('EPOCH : {} , UMAP projection with anomaly {}'.format(epoch_num,anomaly_value), fontsize=18);
    now = datetime.now()
    time = now.strftime("%d-%m-%Y %H %M %S")
    Title = ROUTE+"  {},   Anomaly Value {} , Epoch Num {}.png".format(time,anomaly_value,epoch_num)
    plt.savefig(Title)
    plt.show()

def show_loaded_images(loaded_data):
    real_batch = next(iter(loaded_data))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

global SPLIT   
dataset = datasets.FashionMNIST(root='dataset/',download=True,train=True,transform=transforms)
SPLIT={0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
for i in range(len(dataset.targets)):
    SPLIT[dataset.targets[i].item()].append(i)

global TEST_SPLIT   
dataset = datasets.FashionMNIST(root='dataset/',download=True,train=False,transform=transforms)
TEST_SPLIT={0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
for i in range(len(dataset.targets)):
    TEST_SPLIT[dataset.targets[i].item()].append(i)

def split(data,target): 
    data.data = data.data[SPLIT[target]]
    data.targets = data.targets[SPLIT[target]]

def show_image(dataset,n):
    try: 
        temp_loader = dataloader(dataset, shuffle=True, batch_size = DCGAN_Hyperparameters.batch_size)
        for _ in range(n):
            images = next(iter(temp_loader))
            plt.imshow(images[0].reshape(64,64), cmap="gray")
            plt.show()
    except:
        temp_loader = dataloader(dataset, shuffle=True, batch_size = DCGAN_Hyperparameters.batch_size)
        for _ in range(n):
            images = next(iter(temp_loader))
            plt.imshow(images[0].reshape(64,64).detach().cpu(), cmap="gray")
            plt.show()


class MNIST_Dataset(data.Dataset):
    
    def __init__(self, data, targets, transform):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        if self.transform:
            x = Image.fromarray(x.detach().cpu().numpy())
            x = self.transform(x)
        return x, y

def Just_Add(Generator, Original_data, Size, Target_Value):  
    Z = torch.randn(Size,DCGAN_Hyperparameters.nz,1,1).to(device)
    out_gen = Generator(Z)
    generated_sample = out_gen.reshape(Size,64,64).to(device)
    all_data = torch.cat([Original_data.data, generated_sample],dim=0)
    back_part = torch.tensor([Target_Value for _ in range(Size)])
    all_targets = torch.cat([Original_data.targets, back_part],dim=0).to(device)
    temp = MNIST_Dataset(all_data,all_targets,transforms.ToTensor())
    return temp

def shuffle(Data):
    size = len(Data)
    order = np.random.choice(size,size,replace = False)
    Data.data = Data.data[order]
    Data.targets = Data.targets[order]

def reduce(Data,size):
    order = np.random.choice(len(Data),size,replace = False)
    Data.data = Data.data[order]
    Data.targets = Data.targets[order]

def call_data():
    Numpy_File = torch.tensor(np.load(ROUTE+'\\training_data.npy')).to(device)
    train = datasets.FashionMNIST(root='dataset/', train = True, transform = transforms.ToTensor(), download = True)
    Targets = train.targets
    Dataset = MNIST_Dataset(Numpy_File,Targets,transforms.ToTensor())
    return Dataset

def call_test_data():
    Numpy_File = torch.tensor(np.load(ROUTE+'\\testing_data.npy')).to(device)
    test = datasets.FashionMNIST(root='dataset/', train = False, transform = transforms.ToTensor(), download = True)
    Targets = test.targets
    Dataset = MNIST_Dataset(Numpy_File,Targets,transforms.ToTensor())
    return Dataset

class DCGAN_Hyperparameters:
    nc=1
    batch_size = 128
    image_size = 64
    nz = 100
    ngf = 64
    ndf = 64
    num_epochs = 15
    beta1 = 0.5

UNIQUE_LEARNING_RATES = {0:0.0003, 1: 0.00029, 2: 0.00021, 3: 0.00031,
                         4: 0.00032, 5: 0.00042 , 6: 0.0003 , 
                         7: 0.00014, 8: 0.00032 , 9:0.0002}

class Deep_SVDD_Args:
    num_epochs=100
    num_epochs_ae=100
    lr=0.0001
    weight_decay=0.5e-6
    weight_decay_ae=0.5e-3
    lr_ae=0.0001
    lr_milestones=[50]
    batch_size=200
    pretrain=True
    latent_dim=32


class CNN_Args:
    learning_rate = 3e-4
    num_epochs = 10
    batch_size = 100


TARGETS = [0,2,5,8]
TRAIN_SIZES = [500, 1000, 3000, 6000]
GEN_SIZES = [100, 300, 1000, 3000]

def RUN_CNN(input_data, CNN_Args, device):
    print("\nRunning CNN with {} as anomaly value.".format(CNN_Args.normal_class))
    CNN_Model = Classify_with_CNN(CNN_Args,input_data, device)
    CNN_Model.CNN_train()
    CNN_Model.CNN_test()
    CNN_accuracy = CNN_Model.get_scores()
    print("CNN Accuracy : ",CNN_accuracy)
    return CNN_accuracy

# BASELINE TESTS
Baseline_Results=[]
for small_target in TARGETS:
    print("\n\n")    
    CNN_Args.normal_class = small_target
    
    Test_Data = call_test_data()
    Test_Data.targets = torch.tensor(np.where(Test_Data.targets.detach().cpu()==small_target,0,1)).to(device)
    Test_Loaded = dataloader(Test_Data, shuffle=True, batch_size = DCGAN_Hyperparameters.batch_size)

    ### CNN 은 60000 개 데이터 전체를 사용해 학습을 하지만 Deep SVDD 은 이상치 6000개 가지고만 학습을 한다.
    Train_Baseline = call_data()
    Train_Baseline.targets = torch.tensor(np.where(Train_Baseline.targets.detach().cpu()==small_target,0,1)).to(device)
    Loaded_Baseline = dataloader(Train_Baseline, shuffle = True, batch_size = DCGAN_Hyperparameters.batch_size)
    input_data = [Loaded_Baseline,Test_Loaded]
    CNN_Accuracy=RUN_CNN(input_data,CNN_Args,device)

    ### Deep SVDD
    Deep_SVDD_Args.normal_class = small_target
    Train_Baseline = call_data()
    split(Train_Baseline,small_target)   # 이 한 줄이 Deep SVDD 와 CNN 의 차이.
    Loaded_Baseline = dataloader(Train_Baseline, shuffle = True, batch_size = DCGAN_Hyperparameters.batch_size)
    input_data = [Loaded_Baseline,Test_Loaded]
    ROC_AUC_Scores = RUN_DEEPSVDD(input_data, Deep_SVDD_Args, device)
    Baseline_Results.append([small_target,CNN_Accuracy,ROC_AUC_Scores])


Tests_Results = []

for small_target in TARGETS:
    print("\n\n\n----------------------------------------------------------")
    print("Starting Tests with anomaly value as {}".format(small_target))

    Deep_SVDD_Args.normal_class = small_target
    DCGAN_Hyperparameters.normal_class = small_target

    DCGAN_Hyperparameters.lr = UNIQUE_LEARNING_RATES[small_target]
    CNN_Args.normal_class = small_target

    Test_Data = call_test_data()
    Test_Data.targets = torch.tensor(np.where(Test_Data.targets.detach().cpu()==small_target,0,1)).to(device)
    Test_Loaded = dataloader(Test_Data, shuffle=True, batch_size = DCGAN_Hyperparameters.batch_size)

    for train_size in TRAIN_SIZES:

        print("\n-----------------------------------")
        print("Training DCGAN with {} amount of training data".format(train_size))
        add = "target: {}, size:{} .".format(small_target,train_size)
        Route =  ROUTE+"/content/drive/MyDrive/GAN 딥러닝 연구 관련/Umap/"+add

        Dataset = call_data()
        split(Dataset,small_target)
        reduce(Dataset,train_size)
        Loaded_Data = dataloader(Dataset, shuffle = True, batch_size = DCGAN_Hyperparameters.batch_size)   
        
        DCModel = Run_DCGAN(Loaded_Data,DCGAN_Hyperparameters,device)
        DCModel.train()
        DCModel.show_images(save=False,test_num=0, route = Route +"   ")
        Trained_Generator = DCModel.return_generator()

        for gen_size in GEN_SIZES:
            
            print("Training Deep SVDD with {} amount of generated data".format(gen_size))
            training_data = call_data()
            split(training_data,small_target)  
            added_data = Just_Add(Trained_Generator,training_data,gen_size,small_target)
            train_loader = dataloader(added_data, shuffle=True, batch_size = DCGAN_Hyperparameters.batch_size)
            input_data = [train_loader,Test_Loaded]
            ROC_AUC_Scores = RUN_DEEPSVDD(input_data, Deep_SVDD_Args, device)

            print("Training CNN with {} amount of generated data".format(gen_size))
            training_data = call_data() 
            added_data = Just_Add(Trained_Generator,training_data,gen_size,small_target)
            added_data.targets = torch.tensor(np.where(added_data.targets.detach().cpu()==small_target,0,1)).to(device)
            train_loader = dataloader(added_data, shuffle=True, batch_size = DCGAN_Hyperparameters.batch_size)
            input_data = [train_loader,Test_Loaded]
            CNN_Accuracy=RUN_CNN(input_data,CNN_Args,device)
            Tests_Results.append([small_target,train_size,gen_size,ROC_AUC_Scores,CNN_Accuracy])



