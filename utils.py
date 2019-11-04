import torch

class AddNoise(torch.nn.Module):
    def __init__(self, mean=0.0, std=0.1):
        super(AddNoise, self).__init__()
        self.mean = mean
        self.stddev = std

    def forward(self, input):
        noise = input.clone().normal_(self.mean, self.stddev)
        return input + noise
    
# def hessian_vector_product():
    
#     return None

def hessian_vector_product(loss, model, v):
	grad = torch.autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
	Hv = torch.autograd.grad(grad, model.parameters(), grad_outputs=v, retain_graph=True)
	return Hv