import torch
import torch.nn as nn
import numpy as np
import time

loss_metric = nn.CrossEntropyLoss()

def train_eval_(net, cuda, train_loader, verbose = 1, flatten = True):
    correct = 0
    total = 0
    loss_sum = 0
    bs_i = 0
    for images, labels in train_loader:
        if cuda:
            images, labels = images.cuda(), labels.cuda()
        if flatten:
            images = images.view(-1, 28*28)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.float() == labels.float()).sum()
        loss_sum += loss_metric(outputs,labels)
        bs_i += 1
        
    if verbose:
        print('Train accuracy: %.4f %%' % (100 * np.float(correct) / np.float(total)))

    return (100.0 * correct / total).cpu().data.numpy().item(), loss_sum.cpu().data.numpy().item() / bs_i

def test_eval_(net, cuda, test_loader, verbose = 1, flatten = True):
    correct = 0
    total = 0
    loss_sum = 0
    bs_i = 0
    for images, labels in test_loader:
        if cuda:
            images, labels = images.cuda(), labels.cuda()
        if flatten:
            images = images.view(-1, 28*28)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.float() == labels.float()).sum()
        loss_sum += loss_metric(outputs,labels)
        bs_i += 1
        
    if verbose:
        print('Test accuracy: %.4f %%' % (100 * np.float(correct) / np.float(total)))

    return (100.0 * correct / total).cpu().data.numpy().item(), loss_sum.cpu().data.numpy().item() / bs_i



def train(net, epochs, cuda, optimizer, train_loader, test_loader, flatten=True):

    def train_eval(verbose = 1):
        correct = 0
        total = 0
        loss_sum = 0
        bs_i = 0
        for images, labels in train_loader:
            if cuda:
                images, labels = images.cuda(), labels.cuda()
            if flatten:
                images = images.view(-1, 28*28)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.float() == labels.float()).sum()
            loss_sum += loss_metric(outputs,labels)
            bs_i += 1
            
        if verbose:
            print('Train accuracy: %.4f %%' % (100 * correct / total))
    
        return 100.0 * correct / total, loss_sum.cpu().data.numpy().item() / bs_i
    
    def test_eval(verbose = 1):
        correct = 0
        total = 0
        loss_sum = 0
        bs_i = 0
        for images, labels in test_loader:
            if cuda:
                images, labels = images.cuda(), labels.cuda()
            if flatten:
                images = images.view(-1, 28*28)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.float() == labels.float()).sum()
            loss_sum += loss_metric(outputs,labels)
            bs_i += 1
            
        if verbose:
            print('Test accuracy: %.4f %%' % (100 * correct / total))
    
        return 100.0 * correct / total, loss_sum.cpu().data.numpy().item() / bs_i

    t1 = time.time()

    train_loss_store = []
    test_loss_store = []
    train_acc_store = []
    test_acc_store = []
    
    for epoch in range(epochs):
        for i, (x,y) in enumerate(train_loader):
            if flatten:
                x = x.view(-1,28*28)
            if cuda:
                x = x.cuda()
                y = y.cuda()
    
            optimizer.zero_grad()
            outputs = net.forward(x)
            loss = loss_metric(outputs,y)
            loss.backward()           
            optimizer.step()
    
        print ('Epoch [%d/%d], Loss: %.4f' %(epoch+1, epochs, loss.data.item()))

        #storing training stats:
        train_perc, loss_train = train_eval()
        test_perc, loss_test = test_eval()
        train_loss_store.append(loss_train)
        test_loss_store.append(loss_test)
        train_acc_store.append(train_perc)
        test_acc_store.append(test_perc)

    stats = np.array(train_loss_store), np.array(test_loss_store), np.array(train_acc_store), np.array(test_acc_store)
    t2 = time.time()
    print('Training finished. Time elapsed:',(t2-t1)/60,'minutes')        
    
    return net, stats
    
    # return net, np.array(train_loss_store), np.array(test_loss_store), np.array(train_acc_store), np.array(test_acc_store)














