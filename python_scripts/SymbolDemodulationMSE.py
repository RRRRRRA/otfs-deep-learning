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
BASE_PATH = '/home/jturley1/gokhale_user/675/project/'


# In[4]:


otfs_data = scipy.io.loadmat(BASE_PATH + 'data/OTFSData.mat')
otfs_data.keys()


# In[5]:


Tdata = otfs_data['Tdata'] # transmitted data
Rx = otfs_data['Rx'] # received data
Tdata.shape, Rx.shape


# In[6]:


np.unique(Tdata[:, :, 0][:, 30:])


# In[7]:


n = Rx.shape[0]
indices = np.arange(n)
train_indices, test_indices = train_test_split(indices, test_size = 0.2, shuffle = True, random_state = 42)
train_indices.shape, test_indices.shape


# In[8]:


class SymbolDemodulationDatasetMSE(Dataset):
    def __init__(self, otfs_data, indices, channelEstimationModel = None, ceDiscrete = False):
        self.otfs_data = otfs_data
        self.indices = indices

        self.n_samples = len(self.indices)

        # information symbols received, and predicted channel parameters
        # Multiply by two for real and complex
        self.n_features = 2220 * 2 + 16
        self.n_outputs = 1920 * 2

        self.X = np.zeros((self.n_samples, self.n_features))
        self.Y = np.zeros((self.n_samples, self.n_outputs))
        for i, idx in enumerate(self.indices):
            rx = otfs_data['Rx'][idx]
            
            rx_info = rx[2220:]
            rx_info_real = np.real(rx_info)
            rx_info_imag = np.imag(rx_info)

            self.X[i, :4440] = np.hstack((rx_info_real, rx_info_imag))

            # If no model given, just use truth
            if channelEstimationModel is None:
                chanParams = otfs_data['chanParamsData'][idx]
                numPaths = chanParams[0]['numPaths'][0, 0][0, 0]
                pathDelays = chanParams[0]['pathDelays'][0, 0][0]
                pathGains = chanParams[0]['pathGains'][0, 0][0]
                pathDopplers = chanParams[0]['pathDopplers'][0, 0][0]
            
                pathDelays = np.pad(pathDelays, (0, 5 - len(pathDelays)))
                pathGains = np.pad(pathGains, (0, 5 - len(pathGains)))
                pathDopplers = np.pad(pathDopplers, (0, 5 - len(pathDopplers)))

                self.X[i, 4440:] = np.hstack((numPaths, pathDelays, pathGains, pathDopplers))
                
            # In this case, consider outputs as continuous
            tdata = otfs_data['Tdata'][:, :, idx]
            tdata_info = tdata[:, 30:].reshape(-1)
            tdata_info_real = np.real(tdata_info)
            tdata_info_imag = np.imag(tdata_info)
            self.Y[i] = np.hstack((tdata_info_real, tdata_info_imag))

        self.X = torch.from_numpy(self.X)
        self.Y = torch.from_numpy(self.Y)
    
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# In[9]:


BIT_VALUES = [
    0.7071067811865476+0.7071067811865475j,
    0.7071067811865474-0.7071067811865477j,
    -0.7071067811865475+0.7071067811865476j,
    -0.7071067811865477-0.7071067811865475j
]
class SymbolDemodulationDatasetCE(Dataset):
    def __init__(self, otfs_data, indices, channelEstimationModel = None, ceDiscrete = False):
        self.otfs_data = otfs_data
        self.indices = indices

        self.n_samples = len(self.indices)

        # information symbols received, and predicted channel parameters
        # Multiply by two for real and complex
        self.n_features = 2220 * 2 + 16
        self.n_outputs = 1920

        self.X = np.zeros((self.n_samples, self.n_features))
        self.Y = np.zeros((self.n_samples, self.n_outputs))
        for i, idx in enumerate(self.indices):
            rx = otfs_data['Rx'][idx]
            
            rx_info = rx[2220:]
            rx_info_real = np.real(rx_info)
            rx_info_imag = np.imag(rx_info)

            self.X[i, :4440] = np.hstack((rx_info_real, rx_info_imag))

            # If no model given, just use truth
            if channelEstimationModel is None:
                chanParams = otfs_data['chanParamsData'][idx]
                numPaths = chanParams[0]['numPaths'][0, 0][0, 0]
                pathDelays = chanParams[0]['pathDelays'][0, 0][0]
                pathGains = chanParams[0]['pathGains'][0, 0][0]
                pathDopplers = chanParams[0]['pathDopplers'][0, 0][0]
            
                pathDelays = np.pad(pathDelays, (0, 5 - len(pathDelays)))
                pathGains = np.pad(pathGains, (0, 5 - len(pathGains)))
                pathDopplers = np.pad(pathDopplers, (0, 5 - len(pathDopplers)))

                self.X[i, 4440:] = np.hstack((numPaths, pathDelays, pathGains, pathDopplers))

            # Encode outputs as series of one hot vectors
            tdata = otfs_data['Tdata'][:, :, idx]
            tdata_info = tdata[:, 30:].reshape(-1)
            self.Y[i, :] = np.array([BIT_VALUES.index(t) for t in tdata_info])
            # self.Y[i, :, 0] = (tdata_info == BIT_VALUES[0]).astype(int)
            # self.Y[i, :, 1] = (tdata_info == BIT_VALUES[1]).astype(int)
            # self.Y[i, :, 2] = (tdata_info == BIT_VALUES[2]).astype(int)
            # self.Y[i, :, 3] = (tdata_info == BIT_VALUES[3]).astype(int)

        self.X = torch.from_numpy(self.X)
        self.Y = torch.from_numpy(self.Y.astype(int))
    
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# In[10]:


class SymbolDemodulationNetworkMSE(nn.Module):
    def __init__(self, dropout_rate = 0.0):
        super(SymbolDemodulationNetworkMSE, self).__init__()

        self.lin1 = nn.Linear(in_features=4456, out_features=3840).double()
        self.lin2 = nn.Linear(in_features=3840, out_features=3840).double()
        self.lin3 = nn.Linear(in_features=3840, out_features=3840).double()
        self.lin4 = nn.Linear(in_features=3840, out_features=3840).double()
        self.lin5 = nn.Linear(in_features=3840, out_features=3840).double()

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.dropout4 = nn.Dropout(dropout_rate)

        self.bn1 = nn.BatchNorm1d(3840).double()
        self.bn2 = nn.BatchNorm1d(3840).double()
        self.bn3 = nn.BatchNorm1d(3840).double()
        self.bn4 = nn.BatchNorm1d(3840).double()

    def forward(self, x):
        x = self.dropout1(F.relu(self.bn1(self.lin1(x))))
        x = self.dropout2(F.relu(self.bn2(self.lin2(x))))
        x = self.dropout3(F.relu(self.bn3(self.lin3(x))))
        x = self.dropout4(F.relu(self.bn4(self.lin4(x))))
        x = self.lin5(x)
        return x

class SymbolDemodulationNetworkCE(nn.Module):
    def __init__(self, dropout_rate = 0.0):
        super(SymbolDemodulationNetworkCE, self).__init__()

        self.lin1 = nn.Linear(in_features=4456, out_features=3840).double()
        self.lin2 = nn.Linear(in_features=3840, out_features=3840).double()
        self.lin3 = nn.Linear(in_features=3840, out_features=3840).double()
        self.lin4 = nn.Linear(in_features=3840, out_features=3840).double()

        lin5_ce_layers_list = []
        for i in range(1920):
            l = nn.Linear(in_features=3840, out_features=4).double()
            lin5_ce_layers_list.append(l)
        self.lin5_ce_layers = nn.ModuleList(lin5_ce_layers_list)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.dropout4 = nn.Dropout(dropout_rate)

        self.bn1 = nn.BatchNorm1d(3840).double()
        self.bn2 = nn.BatchNorm1d(3840).double()
        self.bn3 = nn.BatchNorm1d(3840).double()
        self.bn4 = nn.BatchNorm1d(3840).double()

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.dropout1(F.relu(self.bn1(self.lin1(x))))
        x = self.dropout2(F.relu(self.bn2(self.lin2(x))))
        x = self.dropout3(F.relu(self.bn3(self.lin3(x))))
        x = self.dropout4(F.relu(self.bn4(self.lin4(x))))

        # Create tensor to hold all outputs
        y_out = torch.zeros((batch_size, len(self.lin5_ce_layers), 4)).to(x.get_device())
        for i in range(len(self.lin5_ce_layers)):
            y = self.lin5_ce_layers[i](x)
            y_out[:, i, :] = y

        return y_out


# In[11]:


def train_model_ce(train_loader, test_loader, epochs = 50, lr = 1e-4, optimizer_name = 'adam', dropout_rate = 0.0, verbose = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print('Device =', device)

    model = SymbolDemodulationNetworkCE(dropout_rate)
    model = model.to(device)

    if optimizer_name.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer_name.lower() == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    else:
        print('Optimizer not recognized:', optimizer_name)

    loss_fn_ce = nn.CrossEntropyLoss()

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

            # Calculate individual CE losses and sum
            loss = torch.tensor(0.0).to(device)
            for i in range(Y.shape[1]):
                Y_i = Y[:, i]
                Y_hat_i = Y_hat[:, i, :]
                l = loss_fn_ce(Y_hat_i, Y_i)
                loss += l
            loss /= Y.shape[1]

            loss_train.append(loss.item())
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, disable = not verbose):
                X, Y = batch
                X = X.to(device)
                Y = Y.to(device)
                Y_hat = model(X)
    
                # Calculate individual CE losses and sum
                loss = torch.tensor(0.0).to(device)
                for i in range(Y.shape[1]):
                    Y_i = Y[:, i]
                    Y_hat_i = Y_hat[:, i, :]
                    loss += loss_fn_ce(Y_hat_i, Y_i)
                loss /= Y.shape[1]

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
            s = BASE_PATH + 'model_weights/demod_ce_{}_{}_{}.pth'.format(lr, optimizer_name, dropout_rate)
            torch.save(model.state_dict(), s)

        if verbose:
            print('-' * 80)

    return loss_train_means, loss_test_means


# In[12]:


def train_model_mse(train_loader, test_loader, epochs = 50, lr = 1e-4, optimizer_name = 'adam', dropout_rate = 0.0, verbose = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print('Device =', device)

    model = SymbolDemodulationNetworkMSE(dropout_rate)
    model.to(device)

    if optimizer_name.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer_name.lower() == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    else:
        print('Optimizer not recognized:', optimizer_name)

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
            s = BASE_PATH + 'model_weights/demod_mse_{}_{}_{}.pth'.format(lr, optimizer_name, dropout_rate)
            torch.save(model.state_dict(), s)

        if verbose:
            print('-' * 80)

    return loss_train_means, loss_test_means


# In[13]:


lrs = [1e-3, 1e-4, 1e-5]
optimizer_names = ['adam', 'rmsprop', 'adagrad']
dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


try:
    f = open('results_map_symboldemodulation_mse.pkl', 'rb')
    results_map_symboldemodulation_mse = pkl.load(f)
    f.close()
except:
    results_map_symboldemodulation_mse = {}


# In[ ]:


for lr in lrs:
    for optimizer_name in optimizer_names:
        for dropout_rate in dropout_rates:
            dataset_train = SymbolDemodulationDatasetMSE(otfs_data, train_indices)
            train_loader = DataLoader(dataset_train, batch_size = BATCH_SIZE, shuffle=True)
            
            dataset_test = SymbolDemodulationDatasetMSE(otfs_data, test_indices)
            test_loader = DataLoader(dataset_test, batch_size = BATCH_SIZE)

            k = (lr, optimizer_name, dropout_rate)
            print(k)
            if k in results_map_symboldemodulation_mse:
                print(k, 'already trained, skipping')
                continue
            
            results_map_symboldemodulation_mse[k] = train_model_mse(train_loader, test_loader, epochs = EPOCHS,
                           lr = lr, optimizer_name = optimizer_name, dropout_rate = dropout_rate, verbose = True)



            f = open('results_map_symboldemodulation_mse.pkl', 'wb')
            pkl.dump(results_map_symboldemodulation_mse, f)
            f.close()
