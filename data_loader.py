import torch
from torchvision import datasets, transforms

def get_data_loaders(var = 0, mean = 0.05, BS = 128, N_sub = 0):

    transform_data = transforms.ToTensor()
    
    #train set:
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform_data)
    
    if N_sub < 60000 and N_sub != 0:
        train_set.train_data = train_set.train_data[:N_sub]
    
    #adding noise:
    # if noise_level > 0:
    #     train_set.train_data = train_set.train_data.float()
    #     train_set.train_data = train_set.train_data + noise_level*torch.abs(torch.randn(*train_set.train_data.shape))
    #     train_set.train_data = train_set.train_data / train_set.train_data.max()
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = BS, shuffle=False)

    #test set:
    test_set = datasets.MNIST('./data', train=False, download=True, transform=transform_data)
    
    #adding noise to test set:
    if var > 0:
        test_set.test_data = test_set.test_data.float()
        test_set.test_data = test_set.test_data + torch.distributions.Normal(mean,var).sample(test_set.test_data.shape)
        test_set.test_data -= test_set.test_data.min()
        test_set.test_data /= test_set.test_data.max()
        test_set.test_data *= 255
        print(test_set.test_data.min(),test_set.test_data.max())
        # test_set.test_data = test_set.test_data - test_set.test_data.min()
        # test_set.test_data = test_set.test_data / test_set.test_data.max()
        # test_set.test_data = test_set.test_data / test_set.test_data.max()
    
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BS, shuffle=False)
    
    return train_loader, test_loader
    
