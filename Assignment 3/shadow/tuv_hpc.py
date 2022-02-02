from torch import nn
import datetime
import pickle
from sklearn.model_selection import KFold
from plot import *
set_things()

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from torch.utils.data import TensorDataset, DataLoader
from shadow.plot import panel,xlabel,ylabel,legend_on,set_things

class DropoutNormNet(nn.Module):
    def __init__(self,D_in,D_out,layers=[1],dropout_rate=[0.1],batch_norm=True,dropout=True):
        super(DropoutNormNet, self).__init__()
        if len(layers)!=len(dropout_rate):
            dropout_rate = dropout_rate*len(layers)
        self.seq = nn.Sequential()
        for a,b,p,n in zip([D_in]+layers[:-1],layers,dropout_rate,range(1+len(layers))):
            print(n)
            self.seq.add_module("L{}".format(n),nn.Linear(a,b))
            if dropout:
                self.seq.add_module("D{}".format(n),nn.Dropout(p=p))
            self.seq.add_module("A{}".format(n),nn.Tanh())
            if batch_norm:
                self.seq.add_module("BN{}".format(n),nn.BatchNorm1d(b))
        self.seq.add_module("LN",nn.Linear(layers[-1],D_out))

    def forward(self,x):
        return self.seq(x)

def split(X,y,test_size=0.2):
    return [torch.tensor(i).float() for i in train_test_split(X,y,test_size=0.2)]

class DoML():
    def __init__(self,):
        self.model = None
        self.data = None
        self.loss = None
        self.optimizer = None
        self.dataloader_params = {"batch_size":10}

    def set_data(self,data):
        self.Xtrain, self.Xval, self.ytrain, self.yval = data
        self.ytrain = self.ytrain.reshape(-1,1)
        self.yval = self.yval.reshape(-1,1)

    def training(self):
        loss_ = []
        self.model.train()
        dataset = TensorDataset(self.Xtrain,self.ytrain)
        loader = DataLoader(dataset,**self.dataloader_params)
        for batch_idx, data in enumerate(loader):
            x,y = data
            self.optimizer.zero_grad()
            l = self.loss(y,self.model(x).view(y.size())) #.sum()/self.dataloader_params["batch_size"]
            loss_ += [l.item()]
            l.backward()
            self.optimizer.step()
        self.model.eval()
        return np.mean(loss_)

    def cal_loss_val(self):
        self.model.eval()
        l = self.loss(self.yval,self.model(self.Xval).view(self.yval.size()))
        return l.item()

    def cal_loss_train(self):
        self.model.eval()
        l = self.loss(self.ytrain,self.model(self.Xtrain).view(self.ytrain.size()))
        return l.item()

    def train(self,N=10000,n_print=100):
        print(self.model)
        self.train_loss, self.val_loss = [], []
        for epoch in range(N):
            self.training()
            self.train_loss += [self.cal_loss_train()]
            self.val_loss += [self.cal_loss_val()]
            if epoch%n_print==0:
                print("Epoch: ", epoch, "Training loss: ", self.train_loss[-1], "Validation loss: ", self.val_loss[-1])
                self.cal_R2()

        self.model.eval()

    def plot_loss(self,clip0=0,clip1=-1):
        fig, [ax] = panel(1,1)
        ax.plot(range(len(self.train_loss[clip0:clip1])),self.train_loss[clip0:clip1],label="Training")
        ax.plot(range(len(self.val_loss[clip0:clip1])),self.val_loss[clip0:clip1],label="Validation")
        xlabel("Epoch")
        ylabel("MSE Loss")
        legend_on(ax)

    def cal_R2(self,):
        self.model.eval()
        self.R2_val = r2_score(self.yval,self.model(self.Xval).detach().numpy().reshape(self.yval.shape))
        self.R2_train = r2_score(self.ytrain,self.model(self.Xtrain).detach().numpy().reshape(self.ytrain.shape))
        print("Score(val): ",self.R2_val)
        print("Score(train): ",self.R2_train)

logfile='logfile.log'

now = datetime.datetime.now

def myloss(y,ypred):
    return ((y-ypred)**2).sum(axis=0)/y.shape[0]


def dothis(file):
    global Print, print, logfile, layer
    logfile = file+'_log.txt'
    print(file)

    data = np.loadtxt('./'+file.split('_')[0]+'.csv',skiprows=1,delimiter=',')
    train_data, test_data, _, _ = split(data,data)

    np.savetxt('./train_data_'+file+'.csv',train_data)
    np.savetxt('./test_data_'+file+'.csv',test_data)

    data.shape
    train_data.shape
    test_data.shape

    means = train_data.mean(dim=0, keepdim=True)
    stds = train_data.std(dim=0, keepdim=True)
    mask = stds<=0.0001
    stds[mask] = 1
    normalized_data = (train_data - means) / stds
    normalized_test = (test_data - means) / stds

    Xtest = normalized_test[:,:-1]
    ytest = normalized_test[:,-1]

    N = Xtest.shape[0]

    kf = KFold(n_splits=4)
    models = []
    k = 0
    for train_index, test_index in kf.split(normalized_data):
        k+=1
        print("Kfold: ",k)
        Xtrain, Xval = normalized_data[train_index,:-1], normalized_data[test_index,:-1]
        ytrain, yval = normalized_data[train_index,-1], normalized_data[test_index,-1]

        mlc = DoML()
        models.append(mlc)
        mlc.means = means
        mlc.stds = stds
        mlc.set_data([Xtrain, Xval, ytrain, yval])
        drop = False
        norm = False
        if 'drop' in file:
            drop = True
        if 'norm' in file:
            norm = True
        mlc.model = DropoutNormNet(Xtrain.shape[1], 1, layer, [0.1], dropout=drop, batch_norm=norm)
        mlc.loss = myloss
        mlc.optimizer = torch.optim.Adam(mlc.model.parameters(),lr=0.01,weight_decay=0.001)
        mlc.dataloader_params.update({"batch_size":int(N/40),"drop_last":True})


        mlc.train(N=1000)
        m, s = means[0,-1], stds[0,-1]
        print(type(m),type(s),type(ytest))
        mlc.ytest = m+s*ytest
        kkk = mlc.model(Xtest).detach().numpy()
        print(type(kkk))
        mlc.ytest_pred = m+s*torch.from_numpy(kkk)#mlc.model(Xtest).detach().numpy()
        mlc.cal_R2()

        mlc.plot_loss()
        plt.savefig(file+'_loss.png',dpi=300,facecolor='w', edgecolor='k', bbox_inches='tight')
        plt.show()

        plt.hist(((mlc.yval-mlc.model(mlc.Xval))).detach().numpy()*s.item(),bins=50)
        plt.savefig(file+'_error_hist.png',dpi=300,facecolor='w', edgecolor='k', bbox_inches='tight')
        plt.show()
#         print(type(s),type(m),type(mlc.model(mlc.Xval)))
        k2 = mlc.model(mlc.Xval).detach().numpy()
#         print(type(k2))
#         print(k2)
        plt.plot(mlc.yval*s+m,m+s*torch.from_numpy(k2),'o',alpha=0.1)
        lim = [(m+s*mlc.yval).min(),(m+s*mlc.yval).max()]
        plt.plot(lim,lim,)
        plt.savefig(file+'_yvsy_val.png',dpi=300,facecolor='w', edgecolor='k', bbox_inches='tight')
        plt.show()

        plt.plot(mlc.ytest,mlc.ytest_pred,'o',alpha=0.1)
        lim = [mlc.ytest.min(),mlc.ytest.max()]
        plt.plot(lim,lim)
        plt.savefig(file+'_yvsy_test.png',dpi=300,facecolor='w', edgecolor='k', bbox_inches='tight')
        plt.show()
        print('R2(test): ',r2_score(ytest,mlc.model(Xtest).detach().numpy()))



    avg_val_score = np.mean([m.R2_val for m in models])
    avg_train_score = np.mean([m.R2_train for m in models])
    print("Avg val R2: ", avg_val_score)
    print("Avg train R2: ", avg_train_score)
    with open(file+'_models.pickle','bw+') as f:
        pickle.dump(models,f)

layers = {'TUV': [22, 22, 22]}

for file in ['TUV']: #['den','YM','H','SM','TEC','TG','LT','RI']:
    layer = layers[file]
#     print(layer)
    dothis(file+'_drop')
#     dothis(file+'_drop_norm')
