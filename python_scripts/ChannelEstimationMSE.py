#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.io
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pickle as pkl


# In[2]:


torch.cuda.is_available()


# In[3]:


BATCH_SIZE = 32
EPOCHS = 100


# In[4]:


otfs_data = scipy.io.loadmat('OTFSData.mat')
otfs_data.keys()


# In[5]:


Tdata = otfs_data['Tdata'] # transmitted data
Rx = otfs_data['Rx'] # received data
Tdata.shape, Rx.shape


# In[6]:


n = Rx.shape[0]
indices = np.arange(n)
train_indices, test_indices = train_test_split(indices, test_size = 0.2, shuffle = True, random_state = 42)
train_indices.shape, test_indices.shape


# In[7]:


class ChannelEstimationDatasetCEMSE(Dataset):
    def __init__(self, otfs_data, indices):
        self.otfs_data = otfs_data
        self.indices = indices

        self.n_samples = len(self.indices)
        self.n_features = 2220 * 2
        self.n_outputs_channel = 15

        # Generate X and Y matrices 
        self.X = np.zeros((self.n_samples, self.n_features))
        self.Y_paths = np.zeros((self.n_samples))
        self.Y_channel = np.zeros((self.n_samples, self.n_outputs_channel))
        for i, idx in enumerate(self.indices):
            rx = otfs_data['Rx'][idx]
            rx_pilot = rx[:2220]
            rx_pilot_real = np.real(rx_pilot)
            rx_pilot_imag = np.imag(rx_pilot)
            self.X[i] = np.hstack((rx_pilot_real, rx_pilot_imag))
            
            chanParams = otfs_data['chanParamsData'][idx]
            numPaths = chanParams[0]['numPaths'][0, 0][0, 0]
            pathDelays = chanParams[0]['pathDelays'][0, 0][0]
            pathGains = chanParams[0]['pathGains'][0, 0][0]
            pathDopplers = chanParams[0]['pathDopplers'][0, 0][0]
        
            pathDelays = np.pad(pathDelays, (0, 5 - len(pathDelays)))
            pathGains = np.pad(pathGains, (0, 5 - len(pathGains)))
            pathDopplers = np.pad(pathDopplers, (0, 5 - len(pathDopplers)))

            self.Y_paths[i] = numPaths - 1
            self.Y_channel[i] = np.hstack((pathDelays, pathGains, pathDopplers))

        self.X = torch.from_numpy(self.X)
        self.Y_paths = torch.from_numpy(self.Y_paths.astype(int))
        self.Y_channel = torch.from_numpy(self.Y_channel)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.X[idx], (self.Y_paths[idx], self.Y_channel[idx])

class ChannelEstimationDatasetMSE(Dataset):
    def __init__(self, otfs_data, indices):
        self.otfs_data = otfs_data
        self.indices = indices

        self.n_samples = len(self.indices)
        self.n_features = 2220 * 2
        self.n_outputs_channel = 16

        # Generate X and Y matrices 
        self.X = np.zeros((self.n_samples, self.n_features))
        self.Y = np.zeros((self.n_samples, self.n_outputs_channel))
        for i, idx in enumerate(self.indices):
            rx = otfs_data['Rx'][idx]
            rx_pilot = rx[:2220]
            rx_pilot_real = np.real(rx_pilot)
            rx_pilot_imag = np.imag(rx_pilot)
            self.X[i] = np.hstack((rx_pilot_real, rx_pilot_imag))
            
            chanParams = otfs_data['chanParamsData'][idx]
            numPaths = chanParams[0]['numPaths'][0, 0][0, 0]
            pathDelays = chanParams[0]['pathDelays'][0, 0][0]
            pathGains = chanParams[0]['pathGains'][0, 0][0]
            pathDopplers = chanParams[0]['pathDopplers'][0, 0][0]
        
            pathDelays = np.pad(pathDelays, (0, 5 - len(pathDelays)))
            pathGains = np.pad(pathGains, (0, 5 - len(pathGains)))
            pathDopplers = np.pad(pathDopplers, (0, 5 - len(pathDopplers)))

            self.Y[i] = np.hstack((numPaths, pathDelays, pathGains, pathDopplers))

        self.X = torch.from_numpy(self.X)
        self.Y = torch.from_numpy(self.Y)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# In[8]:


class ChannelEstimationNetworkCEMSE(nn.Module):
    def __init__(self, dropout_rate = 0.0):
        super(ChannelEstimationNetworkCEMSE, self).__init__()

        self.lin1 = nn.Linear(in_features=4440, out_features=2048).double()
        self.lin2 = nn.Linear(in_features=2048, out_features=1024).double()
        self.lin3 = nn.Linear(in_features=1024, out_features=512).double()
        self.lin4 = nn.Linear(in_features=512, out_features=256).double()
        self.lin5_sm = nn.Linear(in_features=256, out_features=5).double()
        self.lin5_ch = nn.Linear(in_features=256, out_features=15).double()

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.dropout4 = nn.Dropout(dropout_rate)

        self.bn1 = nn.BatchNorm1d(2048).double()
        self.bn2 = nn.BatchNorm1d(1024).double()
        self.bn3 = nn.BatchNorm1d(512).double()
        self.bn4 = nn.BatchNorm1d(256).double()

    def forward(self, x):
        x = self.dropout1(F.relu(self.bn1(self.lin1(x))))
        x = self.dropout2(F.relu(self.bn2(self.lin2(x))))
        x = self.dropout3(F.relu(self.bn3(self.lin3(x))))
        x = self.dropout4(F.relu(self.bn4(self.lin4(x))))
        x1 = self.lin5_sm(x)
        x2 = self.lin5_ch(x)
        return x1, x2

class ChannelEstimationNetworkMSE(nn.Module):
    def __init__(self, dropout_rate = 0.0):
        super(ChannelEstimationNetworkMSE, self).__init__()

        self.lin1 = nn.Linear(in_features=4440, out_features=2048).double()
        self.lin2 = nn.Linear(in_features=2048, out_features=1024).double()
        self.lin3 = nn.Linear(in_features=1024, out_features=512).double()
        self.lin4 = nn.Linear(in_features=512, out_features=256).double()
        self.lin5 = nn.Linear(in_features=256, out_features=16).double()

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.dropout4 = nn.Dropout(dropout_rate)

        self.bn1 = nn.BatchNorm1d(2048).double()
        self.bn2 = nn.BatchNorm1d(1024).double()
        self.bn3 = nn.BatchNorm1d(512).double()
        self.bn4 = nn.BatchNorm1d(256).double()

    def forward(self, x):
        x = self.dropout1(F.relu(self.bn1(self.lin1(x))))
        x = self.dropout2(F.relu(self.bn2(self.lin2(x))))
        x = self.dropout3(F.relu(self.bn3(self.lin3(x))))
        x = self.dropout4(F.relu(self.bn4(self.lin4(x))))
        x = self.lin5(x)
        return x


# In[9]:


def train_model_ce_mse(train_loader, test_loader, epochs = 50, lr = 1e-4, optimizer_name = 'adam', dropout_rate = 0.0, verbose = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print('Device =', device)

    model = ChannelEstimationNetworkCEMSE(dropout_rate)
    model.to(device)

    if optimizer_name.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer_name.lower() == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    else:
        print('Optimizer not recognized:', optimizer_name)
        return None, None, None

    loss_fn_ce = nn.CrossEntropyLoss()
    loss_fn_mse = nn.MSELoss()

    loss_train_means = []
    loss_train_ce_means = []
    loss_train_mse_means = []
    loss_test_means = []
    loss_test_ce_means = []
    loss_test_mse_means = []

    min_test_loss = float('inf')

    for epoch in range(epochs):
        if verbose:
            print('Epoch', epoch + 1)

        loss_train = []
        loss_train_ce = []
        loss_train_mse = []
        loss_test = []
        loss_test_ce = []
        loss_test_mse = []

        model.train()
        for batch in tqdm(train_loader, disable = not verbose):
            X, (Y_paths, Y_channel) = batch
            X = X.to(device)
            Y_paths = Y_paths.to(device)
            Y_channel = Y_channel.to(device)
            
            Y_hat_paths, Y_hat_channel = model(X)
            loss_ce = loss_fn_ce(Y_hat_paths, Y_paths)
            loss_mse = loss_fn_mse(Y_hat_channel, Y_channel)
            loss_train_ce.append(loss_ce.item())
            loss_train_mse.append(loss_mse.item())
            
            loss = loss_ce + loss_mse
            loss_train.append(loss.item())
            loss.backward()
            optimizer.step()

        # Don't calculate gradients for test set
        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, disable = not verbose):
                X, (Y_paths, Y_channel) = batch
                X = X.to(device)
                Y_paths = Y_paths.to(device)
                Y_channel = Y_channel.to(device)
                
                Y_hat_paths, Y_hat_channel = model(X)
                
                loss_ce = loss_fn_ce(Y_hat_paths, Y_paths)
                loss_mse = loss_fn_mse(Y_hat_channel, Y_channel)
                loss_test_ce.append(loss_ce.item())
                loss_test_mse.append(loss_mse.item())
                
                loss = loss_ce + loss_mse
                loss_test.append(loss.item())

        loss_train_mean = np.mean(loss_train)
        loss_test_mean = np.mean(loss_test)

        loss_train_means.append(loss_train_mean)
        loss_train_ce_means.append(np.mean(loss_train_ce))
        loss_train_mse_means.append(np.mean(loss_train_mse))
        loss_test_means.append(loss_test_mean)
        loss_test_ce_means.append(np.mean(loss_test_ce))
        loss_test_mse_means.append(np.mean(loss_test_mse))

        if verbose:
            print('Train Loss =', loss_train_mean)
            print('Train CE Loss =', np.mean(loss_train_ce))
            print('Train MSE Loss =', np.mean(loss_train_mse))
            print('Test Loss =', loss_test_mean)
            print('Test CE Loss =', np.mean(loss_test_ce))
            print('Test MSE Loss =', np.mean(loss_test_mse))
            
        if loss_test_mean < min_test_loss:
            min_test_loss = loss_test_mean
            if verbose:
                print('New min test loss =', min_test_loss)
            s = 'model_weights/chanest_cemse_{}_{}_{}_{}_{}_{}.pth'.format(epoch + 1, lr, optimizer_name, dropout_rate,
                                                                           loss_train_mean, loss_test_mean)
            torch.save(model.state_dict(), s)

        if verbose:
            print('-' * 80)

    return model, (loss_train_means, loss_train_ce_means, loss_train_mse_means), (loss_test_means, loss_test_ce_means, loss_test_mse_means)


# In[10]:


def train_model_mse(train_loader, test_loader, epochs = 50, lr = 1e-4, optimizer_name = 'adam', dropout_rate = 0.0, verbose = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print('Device =', device)

    model = ChannelEstimationNetworkMSE(dropout_rate)
    model.to(device)

    if optimizer_name.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer_name.lower() == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    else:
        print('Optimizer not recognized:', optimizer_name)
        return None, None, None

    loss_fn_mse = nn.MSELoss()

    loss_train_means = []
    loss_test_means = []

    min_test_loss = float('inf')

    for epoch in range(epochs):
        if verbose:
            print('Epoch', epoch + 1)

        loss_train = []
        loss_test = []

        model.train()
        for batch in tqdm(train_loader, disable = not verbose):
            X, Y = batch
            X = X.to(device)
            Y = Y.to(device)
            
            Y_hat = model(X)
            loss = loss_fn_mse(Y_hat, Y)
            
            loss_train.append(loss.item())
            loss.backward()
            optimizer.step()

        # Don't calculate gradients for test set
        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, disable = not verbose):
                X, Y = batch
                X = X.to(device)
                Y = Y.to(device)
                
                Y_hat = model(X)
                
                loss = loss_fn_mse(Y_hat, Y)
                loss_test.append(loss.item())

        loss_train_mean = np.mean(loss_train)
        loss_test_mean = np.mean(loss_test)
        
        loss_train_means.append(loss_train_mean)
        loss_test_means.append(loss_test_mean)

        if verbose:
            print('Train Loss =', loss_train_mean)
            print('Test Loss =', loss_test_mean)

        if loss_test_mean < min_test_loss:
            min_test_loss = loss_test_mean
            if verbose:
                print('New min test loss =', min_test_loss)
            s = 'model_weights/chanest_mse_{}_{}_{}_{}_{}_{}.pth'.format(epoch + 1, lr, optimizer_name, dropout_rate,
                                                                         loss_train_mean, loss_test_mean)
            torch.save(model.state_dict(), s)

        if verbose:
            print('-' * 80)

    return model, loss_train_means, loss_test_means


# In[ ]:





# In[11]:


lrs = [1e-3, 1e-4, 1e-5]
optimizer_names = ['adam', 'rmsprop', 'adagrad']
dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

try:
    f = open('results_map_mse.pkl', 'rb')
    results_map_mse = pkl.load(f)
    f.close()
except:
    results_map_mse = {}


# In[ ]:


for lr in lrs:
    for optimizer_name in optimizer_names:
        for dropout_rate in dropout_rates:
            dataset_train_mse = ChannelEstimationDatasetMSE(otfs_data, train_indices)
            train_loader_mse = DataLoader(dataset_train_mse, batch_size = BATCH_SIZE, shuffle=True)
            
            dataset_test_mse = ChannelEstimationDatasetMSE(otfs_data, test_indices)
            test_loader_mse = DataLoader(dataset_test_mse, batch_size = BATCH_SIZE)

            k = (lr, optimizer_name, dropout_rate)
            print(k)
            if k in results_map_mse:
                print(k, 'already trained, skipping')
                continue
            results_map_mse[k] = train_model_mse(train_loader_mse, test_loader_mse, epochs = EPOCHS,
                                                lr = lr, optimizer_name = optimizer_name, dropout_rate = dropout_rate, verbose = True)


            f = open('results_map_mse.pkl', 'wb')
            pkl.dump(results_map_mse, f)
            f.close()
