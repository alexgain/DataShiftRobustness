import torch.optim as optim

def get_optimizer(net, lr, opt_str = 'sgd'):
    if opt_str == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.00001)
    elif opt_str == 'sgd-m':
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.8)
    elif opt_str== 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=lr)
    elif opt_str == 'sgd-nes':
        optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=0.001, momentum=0.9, nesterov=True)

    return optimizer