import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse

#directory imports:
from models import MLP
from data_loader import get_data_loaders
from optimizers import get_optimizer
from train import train, train_eval_, test_eval_

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

#experiment-specific args:
# parser.add_argument('--meanMax', default=1.0, type=float, help="select optimizer")

opt = parser.parse_args()

#instantiate model:
net = MLP(input_size = 784, width = opt.netWidth)
if opt.cuda:
    net = net.cuda()
    
#instantiate optimizer:
optimizer = get_optimizer(net = net, lr = opt.lr, opt_str = opt.optim)

#getting data loaders:
train_loader, test_loader = get_data_loaders()

#train model:
net, stats = train(net, opt.epochs, opt.cuda, optimizer, train_loader, test_loader)

#gaussian noise moments:

# max_var = 0.001
# N_var = 5.0
# var = np.arange(0, max_var + max_var/N_var, max_var/N_var)
# mean = 0.05

max_mean = 0.05
N_mean = 10.0
mean = np.arange(0, max_mean + max_mean/N_mean, max_mean/N_mean)
var = 0.001


#getting test loss versus variance of Gaussian noise added:
test_loss_by_var = []
test_acc_by_var = []

# for k in range(var.shape[0]):
    # train_loader, test_loader = get_data_loaders(var = var[k], mean = mean)
for k in range(mean.shape[0]):
    train_loader, test_loader = get_data_loaders(var = var, mean = mean[k])

    test_acc, test_loss = test_eval_(net, opt.cuda, test_loader, verbose = 0)
    test_loss_by_var.append(test_loss)
    test_acc_by_var.append(test_acc)
       
test_loss_by_var = np.array(test_loss_by_var)
test_acc_by_var = np.array(test_acc_by_var)

# test_loss_by_var = np.insert(test_loss_by_var,0,stats[1][-1])
# var = np.insert(var,0,0)

#plotting results:
fig, ax = plt.subplots()
try:
    ax.plot(var,test_loss_by_var,'-o',color='blue')
except:
    ax.plot(mean,test_loss_by_var,'-o',color='blue')
plt.title('Test Loss Versus Mean of Added Gaussian Noise')
plt.xlabel('Mean')
plt.ylabel('Sample-Average Test Loss')
ax.set_facecolor('lavender')
ax.grid(color='w', linestyle='-', linewidth=2)
plt.savefig('plots/loss_vs_var.png',dpi=100)
plt.show()

#plotting results:
# fig, ax = plt.subplots()
# try:
#     ax.plot(var,test_acc_by_var,'-o',color='blue')
# except:
#     ax.plot(mean,test_acc_by_var,'-o',color='blue')
# plt.title('Test Acc Versus Variance of Added Gaussian Noise')
# plt.xlabel('Mean')
# plt.ylabel('Test Accuracy (%)')
# ax.set_facecolor('lavender')
# ax.grid(color='w', linestyle='-', linewidth=2)
# plt.savefig('plots/acc_vs_var.png',dpi=100)
# plt.show()



