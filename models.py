import torch.nn as nn


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
        
    def forward(self, x, bn = False):

        if not bn:        
            x = self.relu(self.ff1(x))
            x = self.relu(self.ff2(x))
            x = self.relu(self.ff3(x))        
        elif bn:
            x = self.relu(self.bn1(self.ff1(x)))
            x = self.relu(self.bn2(self.ff2(x)))
            x = self.relu(self.bn3(self.ff3(x)))            

        x = self.ff_out(x)            
            
        return x



        