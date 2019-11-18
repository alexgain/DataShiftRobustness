import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, input_size=784, width=500, num_classes=10):
        super(MLP, self).__init__()
        self.ff1 = nn.Linear(input_size, width)
        self.ff2 = nn.Linear(width, width) 
        self.ff3 = nn.Linear(width, width)
        self.ff_out = nn.Linear(width, num_classes)
        
        ##activations:
        self.do = nn.Dropout()
        self.relu = nn.ReLU()
        self.sm = nn.Softmax()
        
        ##BN:
        self.bn1 = nn.BatchNorm1d(width)
        self.bn2 = nn.BatchNorm1d(width)
        self.bn3 = nn.BatchNorm1d(width)
        
        ##noise parameters (just to include as part of computational graph):
        self.mean = nn.Parameter(torch.zeros(1),requires_grad=True)
        self.var = nn.Parameter(torch.zeros(1),requires_grad=True)
        
    def forward(self, x):

        # noise = x.clone().normal_(self.mean, torch.sqrt(self.var))
        # x = x + noise
        x = self.relu(self.bn1(self.ff1(x)))
        x = self.relu(self.bn2(self.ff2(x)))
        x = self.relu(self.bn3(self.ff3(x)))            

        x = self.ff_out(x)            
            
        return x



        