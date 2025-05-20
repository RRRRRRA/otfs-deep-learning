from tqdm import tqdm
import glob
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.io
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pickle as pkl

BATCH_SIZE = 32
EPOCHS = 100
BASE_PATH = '/home/jturley1/gokhale_user/675/project/'

otfs_data = scipy.io.loadmat(BASE_PATH + 'data/OTFSData.mat')
otfs_data.keys()

Tdata = otfs_data['Tdata'] # transmitted data
Rx = otfs_data['Rx'] # received data

n = Rx.shape[0]
indices = np.arange(n)
train_indices, test_indices = train_test_split(indices, test_size = 0.2, shuffle = True, random_state = 42)

class SymbolDemodulationDatasetCNNMSE(Dataset):
    def __init__(self, otfs_data, indices, channelEstimationModel = None, ceDiscrete = False):
        self.otfs_data = otfs_data
        self.indices = indices

        self.n_samples = len(self.indices)

        # information symbols received, and predicted channel parameters
        # Multiply by two for real and complex
        #self.n_features = 2220 * 2 + 16
        #self.n_outputs = 1920 * 2

        # self.X = np.zeros((self.n_samples, 2, 74, 30))
        self.X = np.zeros((self.n_samples, 2, 128, 128))
        # self.Y = np.zeros((self.n_samples, 2, 64, 30))
        self.Y = np.zeros((self.n_samples, 2, 64, 64))
        for i, idx in enumerate(self.indices):
            rx = otfs_data['Rx'][idx]
            
            rx_info = rx[2220:].reshape(74, 30)
            rx_info_real = np.real(rx_info)
            rx_info_imag = np.imag(rx_info)

            self.X[i, 0, 27:(128-27), 49:(128-49)] = rx_info_real
            self.X[i, 1, 27:(128-27), 49:(128-49)] = rx_info_imag

            # If no model given, just use truth
            # if channelEstimationModel is None:
            #     chanParams = otfs_data['chanParamsData'][idx]
            #     numPaths = chanParams[0]['numPaths'][0, 0][0, 0]
            #     pathDelays = chanParams[0]['pathDelays'][0, 0][0]
            #     pathGains = chanParams[0]['pathGains'][0, 0][0]
            #     pathDopplers = chanParams[0]['pathDopplers'][0, 0][0]
            
            #     pathDelays = np.pad(pathDelays, (0, 5 - len(pathDelays)))
            #     pathGains = np.pad(pathGains, (0, 5 - len(pathGains)))
            #     pathDopplers = np.pad(pathDopplers, (0, 5 - len(pathDopplers)))

            #     self.X[i, 4440:] = np.hstack((numPaths, pathDelays, pathGains, pathDopplers))
                
            # In this case, consider outputs as continuous
            tdata = otfs_data['Tdata'][:, :, idx]
            tdata_info = tdata[:, 30:]
            tdata_info_real = np.real(tdata_info)
            tdata_info_imag = np.imag(tdata_info)
            self.Y[i, 0, :, 17:(64-17)] = tdata_info_real
            self.Y[i, 1, :, 17:(64-17)] = tdata_info_imag

        self.X = torch.from_numpy(self.X)
        self.Y = torch.from_numpy(self.Y)
    
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class SymbolDemodulationNetworkCNNMSE(nn.Module):
    def __init__(self, dropout_rate = 0.0):
        super(SymbolDemodulationNetworkCNNMSE, self).__init__()

        self.conv1 = nn.Conv2d(2, 32, 3, padding = 1, stride = 1).double()
        self.conv2 = nn.Conv2d(32, 32, 3, padding = 1, stride = 1).double()
        self.conv3 = nn.Conv2d(32, 32, 3, padding = 1, stride = 2).double()
        self.conv4 = nn.Conv2d(32, 32, 3, padding = 1, stride = 1).double()
        self.conv5 = nn.Conv2d(32, 2, 3, padding = 1, stride = 1).double()

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.dropout4 = nn.Dropout(dropout_rate)

        self.bn1 = nn.BatchNorm2d(32).double()
        self.bn2 = nn.BatchNorm2d(32).double()
        self.bn3 = nn.BatchNorm2d(32).double()
        self.bn4 = nn.BatchNorm2d(32).double()

    def forward(self, x):
        x = self.dropout1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout4(F.relu(self.bn4(self.conv4(x))))
        x = self.conv5(x)
        return x

def train_model_cnn_mse(train_loader, test_loader, epochs = 50, lr = 1e-4, optimizer_name = 'adam', dropout_rate = 0.0, verbose = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print('Device =', device)

    model = SymbolDemodulationNetworkCNNMSE(dropout_rate)
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
            s = BASE_PATH + 'model_weights/demod_cnn_mse_{}_{}_{}_{}_{}_{}.pth'.format(epoch + 1, lr, optimizer_name, dropout_rate,
                                                                       loss_train_mean, loss_test_mean)
            torch.save(model.state_dict(), s)

        if verbose:
            print('-' * 80)

    return loss_train_means, loss_test_means

lrs = [1e-3, 1e-4, 1e-5]
optimizer_names = ['adam', 'rmsprop', 'adagrad']
dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

try:
    f = open(BASE_PATH + 'results_maps/results_map_symboldemodulation_cnn_mse.pkl', 'rb')
    results_map_symboldemodulation_cnn_mse = pkl.load(f)
    f.close()
except:
    results_map_symboldemodulation_cnn_mse = {}

for lr in lrs:
    for optimizer_name in optimizer_names:
        for dropout_rate in dropout_rates:
            dataset_train = SymbolDemodulationDatasetCNNMSE(otfs_data, train_indices)
            train_loader = DataLoader(dataset_train, batch_size = BATCH_SIZE, shuffle=True)
            
            dataset_test = SymbolDemodulationDatasetCNNMSE(otfs_data, test_indices)
            test_loader = DataLoader(dataset_test, batch_size = BATCH_SIZE)

            k = (lr, optimizer_name, dropout_rate)
            print(k)
            if k in results_map_symboldemodulation_cnn_mse:
                print(k, 'already trained, skipping')
                continue

            results_map_symboldemodulation_cnn_mse[k] = train_model_cnn_mse(train_loader, test_loader, epochs = EPOCHS,
               lr = lr, optimizer_name = optimizer_name, dropout_rate = dropout_rate, verbose = True)

            f = open(BASE_PATH + 'results_maps/results_map_symboldemodulation_cnn_mse.pkl', 'wb')
            pkl.dump(results_map_symboldemodulation_cnn_mse, f)
            f.close()
