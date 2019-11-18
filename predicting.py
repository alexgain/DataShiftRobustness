import torch
import torch.nn as nn
import numpy as np
import time
from utils import AddNoise

from train import train_eval_, test_eval_

def get_predictions(net, var, train_loader, test_loader, opt, flatten=True, eps=1e-2):
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

            if opt.prediction == 'alaa': #version with x-sampling included
                
                x.requires_grad = True
                outputs = net.forward(x)
                loss_orig = loss_metric(outputs,y)
                phi_S = torch.stack([torch.zeros(1),torch.ones(1)*var_cur]).view(-1)

                #Shifting:
                mean_S = phi_S[0]
                var_S = phi_S[1]
                mean_S.requires_grad=True
                var_S.requires_grad=True
                x = x + (mean_S + torch.randn(x.shape)*torch.sqrt(var_S))
                                
                outputs = net.forward(x)
                loss_new = loss_metric(outputs,y)

                #get gradient w.r.t. shift parameters at 0:
                grad_phi_S = torch.stack(torch.autograd.grad(loss_new, [mean_S, var_S], retain_graph=True)).view(-1)               
                first_order = grad_phi_S.dot(phi_S)

                if first_order != first_order: #nan check
                    first_order = 0
                # else:
                #     first_order = torch.abs(first_order)
                pred_loss = loss_orig + first_order

            if opt.prediction == 'alaa2': #second order version
                
                x.requires_grad = True
                outputs = net.forward(x)
                loss_orig = loss_metric(outputs,y)
                
                phi_S = torch.stack([torch.zeros(1),torch.ones(1)*var_cur]).view(-1)
                phi_S.requires_grad = True

                #Shifting:
                x = x + (phi_S[0] + torch.randn(x.shape)*torch.sqrt(phi_S[1]))
                
                outputs = net.forward(x)
                loss_new = loss_metric(outputs,y)

                #get gradient w.r.t. shift parameters at phi_S:
                grad_phi_S = torch.autograd.grad(loss_new, phi_S, retain_graph=True)[0].view(-1)
                first_order = grad_phi_S.dot(phi_S)
                if first_order != first_order: #nan check
                    first_order = 0
                    
                #get hessian w.r.t. shift parameters at phi_S:
                grad_phi_S = torch.autograd.grad(loss_new, phi_S, create_graph=True, retain_graph=True)[0]
                H_phi_S = torch.autograd.grad(grad_phi_S, phi_S, grad_outputs=phi_S, retain_graph=True)[0]
                second_order = phi_S.dot(H_phi_S)
                if second_order != second_order: #nan check
                    second_order = 0
                
                #debugging:
                if i == 0:
                    print('var cur',var_cur)
                    print('phi_S[1]',phi_S[1])
                    print('phi_S norm',phi_S.norm())
                    print('grad phi_S norm',grad_phi_S.norm()) #this get smaller as variance increases for some reason
                    print('first order term',first_order)
                    print('second order term',second_order)
                    print()
                
                pred_loss = loss_orig + first_order + 0.5*second_order

                
            if opt.prediction == 'alaa_0': #T.S. expansion at phi = 0 instead
                
                x.requires_grad = True
                phi_S = torch.stack([torch.zeros(1),torch.ones(1)*var_cur]).view(-1)

                #"0"-shift prediction params: (needed so we can get a gradient at shift=0)
                # eps = 1e-8
                mean_0 = torch.zeros(1) + eps
                var_0 = torch.zeros(1) + eps
                mean_0.requires_grad=True
                var_0.requires_grad=True
                x = x + (mean_0 + torch.randn(x.shape)*torch.sqrt(var_0))
                
                outputs = net.forward(x)
                loss_orig = loss_metric(outputs,y)

                #get gradient w.r.t. shift parameters at 0:
                grad_phi_0 = torch.stack(torch.autograd.grad(loss_orig, [mean_0, var_0], retain_graph=True)).view(-1)               
                pred_loss = (loss_orig + torch.abs(grad_phi_0.dot(phi_S)))/opt.batchSize

            if opt.prediction == 'alaa_analytic': #version where x is not used

                x.requires_grad = True
                outputs_orig = net.forward(x)
                loss_orig = loss_metric(outputs_orig,y)                
                outputs_orig = torch.nn.Softmax()(outputs_orig).mean(dim=0).view(1,-1)
                
                phi_S = torch.stack([torch.zeros(1),torch.ones(1)*var_cur]).view(-1)
                phi_S.requires_grad = True
                
                #expected v. of f(delta):
                delta_ev = torch.randn(x.shape)*torch.sqrt(phi_S[1])
                outputs = torch.nn.Softmax()(net.forward(delta_ev)).mean(dim=0).view(1,-1) 
                print(net.forward(delta_ev).mean(dim=0).view(1,-1).norm())
                
                #get gradient w.r.t. shift parameters at 0:
                grad_phi_S_store = []
                for k in range(outputs_orig.shape[1]):
                    grad_phi_S = torch.autograd.grad(outputs[0][k], phi_S, create_graph=True, retain_graph=True)[0]                    
                    grad_phi_S_store.append(grad_phi_S)
                first_order_outputs = torch.stack([grad_phi_S_store[k].dot(phi_S) for k in range(len(grad_phi_S_store))]).view(1,-1)
                first_order_loss_change = (outputs_orig[0]* torch.log(outputs_orig[0] + first_order_outputs[0])).sum() * (-1)
                first_order = first_order_loss_change
                
                # first_order_outputs = grad_phi_S.dot(phi_S)
                if first_order != first_order: #nan check
                    first_order = 0
                pred_loss = loss_orig + first_order

                
            est_loss += pred_loss#/opt.batchSize
        est_loss /= i+1
        predictions.append(est_loss.cpu().data.numpy().item())
    
    return np.array(predictions)







