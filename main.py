import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
import time

#directory imports:
from models import MLP
from data_loader import get_data_loaders
from optimizers import get_optimizer
from train import train, train_eval_, test_eval_
from predicting import get_predictions
from utils import AddNoise

#argument parser:
parser = argparse.ArgumentParser()

#basic args:
parser.add_argument('--dataset', required=False, help='current options: mnist', default='mnist')
parser.add_argument('--dataroot', required=False, help='path to dataset', default='./data')
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--netWidth', default=500, type=int, help="network hidden layer size")
parser.add_argument('--epochs', type=int, default=3, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--optim', default='sgd-m', type=str, help="select optimizer")
parser.add_argument('--cuda', action='store_true', help='enables cuda', default=False)
parser.add_argument('--verbose', action='store_true', help='enables verbosity', default=False)
parser.add_argument('--prediction', type=str, default='vanilla', help='string for prediction method. \
                    current options: vanilla: direct loss computation;\
                    gradient: loss estimation via gradient;\
                    koh: loss estimation via Koh et. al 2017;\
                    alaa: loss estimation via Alaa et. al 2019 (not implemented);\
                    ')
parser.add_argument('--Ns', type=int, default=160, help='number of shifted samples used for loss estimation (for grad methods).')
parser.add_argument('--model_path', type=str, default='', help='path to saved model if loading.')

opt = parser.parse_args()

#instantiate model:
net = MLP(input_size = 784, width = opt.netWidth)
if opt.cuda:
    net = net.cuda()
if opt.model_path != '':
    net.load_state_dict(torch.load(opt.model_path))
    
#instantiate optimizer:
optimizer = get_optimizer(net = net, lr = opt.lr, opt_str = opt.optim)

#getting data loaders:
train_loader, test_loader = get_data_loaders(BS = opt.batchSize)

#train model:
if opt.model_path == '':
    net, stats = train(net, opt.epochs, opt.cuda, optimizer, train_loader, test_loader)
# net, stats = train(torch.nn.Sequential(AddNoise(mean=0,std=np.sqrt(0.25)),net), opt.epochs, opt.cuda, optimizer, train_loader, test_loader)

#gaussian noise moments:

max_var = 2.5
N_var = 10.0
var = np.arange(0, max_var + max_var/N_var, max_var/N_var)
mean = 100

t1 = time.time()
print("Generating shifted data and ground-truth loss...")

#getting test loss versus variance of Gaussian noise added:
test_loss_by_var = []
test_acc_by_var = []


for k in range(var.shape[0]):    
    net_noisy = torch.nn.Sequential(AddNoise(mean=0,std=np.sqrt(var[k])),net)
    test_acc, test_loss = test_eval_(net_noisy, opt.cuda, test_loader, verbose = 0)
    test_loss_by_var.append(test_loss)
    test_acc_by_var.append(test_acc)
    
# for k in range(mean.shape[0]):
#     train_loader, test_loader = get_data_loaders(var = var, mean = mean[k])
#     loaders.append([train_loader,test_loader])
#     test_acc, test_loss = test_eval_(net, opt.cuda, test_loader, verbose = 0)
#     # test_acc, test_loss = train_eval_(net, opt.cuda, test_loader, verbose = 0)
#     test_loss_by_var.append(test_loss)
#     test_acc_by_var.append(test_acc)
       
test_loss_by_var = np.array(test_loss_by_var)
test_acc_by_var = np.array(test_acc_by_var)

t2 = time.time()
print("Data-shifting finished. Time elapsed:",(t2-t1)/60,'minutes')

#getting predictions:
t1 = time.time()
print()
print("Starting predictions...")
predictions = get_predictions(net, var, train_loader, test_loader, opt)
t2 = time.time()
print("Predictions finished. Time elapsed:",(t2-t1)/60,'minutes')

if opt.verbose:
    #plotting results:
    fig, ax = plt.subplots()
    try:
        ax.plot(var,test_loss_by_var,'-o',color='blue', label = 'actual')
        ax.plot(var,predictions,'-o',color='orange', label = 'prediction')
    except:
        ax.plot(mean,test_loss_by_var,'-o',color='blue', label = 'actual')
        ax.plot(mean, predictions,'-o',color='orange', label = 'prediction')
    plt.title('Test Loss Versus Variance of Added Gaussian Noise')
    plt.xlabel('Variance')
    plt.ylabel('Sample-Average Test Loss')
    ax.set_facecolor('lavender')
    ax.grid(color='w', linestyle='-', linewidth=2)
    plt.legend()
    plt.savefig('plots/loss_vs_variance.png',dpi=100)
    plt.show()    
    
