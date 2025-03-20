import torch 
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torchvision
import torch.nn as nn
import torch.optim as optim
"""import matplotlib.pyplot as plt"""
from PIL import Image


from model import Generator, Discriminator, Autoencoder
from utils import D_train, G_train, AE_train, save_models, VAE, VD, VG , VGgpt,afficher_image, afficher_image_random, print_loss
from FID_yanis import create_subset, calculate_fid
from gpt_visu_latent import plotespace


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
    map_location=torch.device('cpu')
print(f'Using device: {device}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GAN.')
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.0003,
                      help="The learning rate to use for training.")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Size of mini-batches for SGD")

    args = parser.parse_args()


    os.makedirs('chekpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    #nouveau 
    os.makedirs('gif_frames', exist_ok=True) # # Dossier pour stocker les étapes intermédiaires ajout ????,,????????????????????
    
   

    
    # Data Pipeline
    print('Dataset loading...')
    # MNIST Dataset
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5))])

    train_dataset = datasets.MNIST(root='data/MNIST/', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='data/MNIST/', train=False, transform=transform, download=False)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=args.batch_size, shuffle=False)
    print('Dataset Loaded.')


    print('Model Loading...')
    mnist_dim = 784
    G = torch.nn.DataParallel(Generator(g_output_dim = mnist_dim)).to(device)
    D = torch.nn.DataParallel(Discriminator(mnist_dim)).to(device)
    AE = torch.nn.DataParallel(Autoencoder(encoder_input_dim = mnist_dim, generator = G)).to(device)
    E = AE.module.encoder
    
    
    print('Model loaded.')

    # define optimizers
    
    G_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)
    D_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)  # Réduction du lr pour le discriminateur


    AE_optimizer = torch.optim.Adam(AE.parameters(), lr = 0.0001)
    AE_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(AE_optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    """G_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(G_optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    D_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(D_optimizer, mode='min', factor=0.5, patience=5, verbose=True)"""



    print('Start Training :')
    
    n_epoch = args.epochs
    looss_G =[]
    looss_D = []
    fid = []
    looss_AE = []
    z = torch.randn(args.batch_size, 100).to(device)
    
    
    fixed_z_samples = []
    for epoch in trange(1, n_epoch+1, leave=True):
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, mnist_dim)
            x = x.to(device)
            
            AE_loss = AE_train(x, G, E, AE_optimizer, VAE)
            looss_AE.append(AE_loss)
            
            D_loss = D_train(x, G, E, D, D_optimizer, VD)
            looss_D.append(D_loss)
            
            G_loss = G_train(x, G, D, G_optimizer, VG)
            looss_G.append(G_loss)  
        """AE_scheduler.step(AE_loss)"""
        
        if epoch % 3 == 0:
            with torch.no_grad():
                G.eval()
                samples = G(z).reshape(-1, 1, 28, 28)  # Générer 25 images
                fixed_z_samples.append(samples.cpu())  # Sauvegarder pour le GIF
                grid = torchvision.utils.make_grid(samples, nrow=5, normalize=True)
                torchvision.utils.save_image(grid, f'gif_frames/epoch_{epoch}.png')
                
            save_models(G, D, AE, 'checkpoints')  
            
            """print_loss(looss_G, looss_D, looss_AE)
            
            afficher_image(x, G,E)
            afficher_image_random(G,E, device)  """ 
            """ 
            plotespace(E, G, 500, 'tsne')
                     """
        save_models(G, D, AE, 'checkpoints')
                
    print('Training done')

        
