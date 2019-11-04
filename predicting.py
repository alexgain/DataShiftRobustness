import torch
import torch.nn as nn
import numpy as np
import time
from utils import AddNoise

from train import train_eval_, test_eval_



def get_predictions(net, var, train_loader, test_loader, opt, flatten=True):
    def bw_dot(z1,z2):
        z3 = torch.ones(z1.shape[0])
        if opt.cuda: z3 = z3.cuda()
        for k1 in range(z1.shape[0]):
            z3[k1] = torch.dot(z1[k1],z2[k1])
        return z3
    loss_metric = nn.CrossEntropyLoss()
    predictions = []
    for var_cur in range(len(var)):
        est_loss = 0
        for i, (x,y) in enumerate(test_loader):
            if (opt.batchSize * i) >= opt.Ns:
                break            
            
            if flatten:
                x = x.view(-1,28*28)
            if opt.cuda:
                x = x.cuda()
                y = y.cuda()
            
            if opt.prediction == 'vanilla':
                net_noisy = torch.nn.Sequential(AddNoise(mean=0,std=np.sqrt(var_cur)),net)
                outputs = net_noisy.forward(x)
                pred_loss = loss_metric(outputs,y)
    
            if opt.prediction == 'gradient':
                    
                x.requires_grad = True
                outputs = net.forward(x)
                loss_orig = loss_metric(outputs,y)
                delta = AddNoise(mean=0,std=np.sqrt(var_cur))(x) - x
                grad_x = torch.autograd.grad(loss_orig, x)[0]
                pred_loss = loss_orig + torch.abs(bw_dot(grad_x,delta)).sum()
                

            if opt.prediction == 'koh':
                    
                x.requires_grad = True
                outputs = net.forward(x)
                loss_orig = loss_metric(outputs,y)
                delta = AddNoise(mean=0,std=np.sqrt(var_cur))(x) - x
                
                #1st order approximation:                                
                grad_x = torch.autograd.grad(loss_orig, x, retain_graph=True)[0]
                # pred_loss = loss_orig + torch.bmm(grad_x.view(opt.batchSize,1,-1),delta.view(opt.batchSize,-1,1)).mean()
                first_order = loss_orig + torch.abs(bw_dot(grad_x,delta)).sum()
                #2nd order term:
                grad_x = torch.autograd.grad(loss_orig, x, retain_graph=True, create_graph=True)[0]
                Hv = torch.autograd.grad(grad_x, x, grad_outputs=delta, retain_graph=True)[0]
                second_order = bw_dot(delta,Hv).mean()
                pred_loss = first_order + 0.5 * second_order
                
            if opt.prediction == 'alaa':
                
                pred_loss = 0

                
            est_loss += pred_loss#/opt.batchSize
        est_loss /= i+1
        predictions.append(est_loss.cpu().data.numpy().item())
    
    return np.array(predictions)







