import torch
import os
import numpy as np
"""import matplotlib.pyplot as plt"""
import torch.nn.utils as nn_utils

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
    map_location=torch.device('cpu')


def D_train(x, G, E, D, D_optimizer, VD):
    #=======================Train the discriminator=======================#
    
    # but maximiser VAE(E,G)
    D.zero_grad()
    D.train()  # Mettre le discriminateur en mode entraînement
    
    x = x.to(device)
    G = G.to(device)
    D = D.to(device)
    E = E.to(device)
    
    # Calcul de la perte V_D
    D_loss = -VD(D, G, E, x)  # On minimise -V_D pour maximiser V_D
    
    # Mise à jour des poids de D
    D_optimizer.zero_grad()
    D_loss.backward()
    """nn_utils.clip_grad_norm_(D.parameters(), 1)"""
    D_optimizer.step()

    return D_loss.data.item() 
    
   
 
def G_train(x, G, D, G_optimizer, VG):
    #=======================Train the generator=======================#
    G.zero_grad()
    G.train()  
    x = x.to(device)
    G = G.to(device)
    D = D.to(device)
    G_loss = VG(D, G, x)  
    G_optimizer.zero_grad()
    G_loss.backward()
    """nn_utils.clip_grad_norm_(G.parameters(), 1)"""
    G_optimizer.step()

    return G_loss.data.item() 



def AE_train(x, G, E, AE_optimizer, VAE):
    """G.zero_grad()
    E.zero_grad()"""
    E.train()  
    G.train()  
    x = x.to(device)
    G = G.to(device)
    E = E.to(device)
    VAE_loss = VAE(E, G, x)  
    
    AE_optimizer.zero_grad()
    VAE_loss.backward()
    AE_optimizer.step()

    return VAE_loss.data.item() 


def l2_norm(x):
    return torch.sqrt(torch.sum(x**2))

def l1_norm(x):
    return torch.sum(torch.abs(x))

def probaconditionnelle(z, sigma): 
    
    # Calculer les distances au carré entre chaque paire de points
    distances = torch.cdist(z, z, p=2) ** 2 # matrice n*n symetrique 0 diago et distances i j = ||zi - zj||^2
    
    affinities = (1 + distances / (2 * sigma**2))**-1 # matrice n*n symetrique 0 diago et affinities[i,j] = ( 1 + ||zi - zj||^2-2sigma ) **-1
    
    affinities.fill_diagonal_(0)
    
    p_cond = affinities / affinities.sum(dim=1, keepdim=True)
    
    return p_cond # preuve voir cahier 


def probajointe(p_cond):
    
    n= p_cond.shape[0] 
    p_joint = (p_cond + p_cond.T) /(2*n) 
    
    return p_joint # a prouver !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def variance(x):
    distances = torch.cdist(x, x, p=2) ** 2 

   
    mean_distance = torch.mean(distances)

    # var = E(X-E(X))^2 
    variance = torch.mean((distances - mean_distance) ** 2)
    return variance



def VR(E,G, x): 
    z = E(x) # marche pas ouf
    sigma_z = variance(z)
    sigma_x = variance(x)

    p_cond = probaconditionnelle(z, sigma_z)
    p_joint = probajointe(p_cond)

 
    G_z = G(z)
    q_cond = probaconditionnelle(G_z, sigma_x)
    q_joint = probajointe(q_cond)
    p_joint = p_joint.clamp(min=1e-8) #pas de 0
    q_joint = q_joint.clamp(min=1e-8)

    kl_div = torch.sum(p_joint * torch.log(p_joint/(q_joint))) 
    return kl_div

    

def VAE(E,G,x, lambdar = 0.5): 
    val1 = l2_norm(x-G(E(x))) 
    val2 = VR(E,G,x)
    val = val1**2 + lambdar*val2 
    return val
    
def VP(D,G,x):
    mu = torch.rand(1, device=device) 
    z = torch.randn(x.shape[0], 100).to(device)
    xchapeau = mu*x + (1-mu)*G(z)
    val = esperance((l2_norm(gradient(D,xchapeau))-1)**2)
    return val
    
def esperance(x):
    return torch.mean(x)

def VD(D,G,E,x,alpha=0.0005,lambdap=0.5): 
    z = torch.randn(x.shape[0], 100).to(device)
    G_z = G(z)
    val1 = (1-alpha)*esperance(torch.log(D(x)))
    val2 = alpha*esperance(torch.log(D(G(E(x)))))
    val3 = esperance(torch.log(1-D(G_z)))
    val4 = VP(D,G,x)
    val = val1 + val2 + val3 - lambdap*val4
    return val
  

def VG(D,G,x, lambda1 = 0.5, lambda2 = 0.5):
    z = torch.randn(x.shape[0], 100).to(device)
    G_z = G(z)
    #D(x) = vecteur de score dim batchsize * 1
    bonneval1 = l1_norm(esperance(D(x))-esperance(D(G_z)))
    grad_dx = gradient(D,x)
    grad_dgz = gradient(D,G_z)
    """bonneval2 =  ( l2_norm ( esperance ((grad_dx))  - esperance(grad_dgz) ) ) ** 2
    bonneval3 =  ( l2_norm ( esperance ((grad_dx.T@x))  - esperance(grad_dgz.T@x) ) ) ** 2"""
    bonneval2 =  ( l2_norm ( esperance (l2_norm(grad_dx))  - esperance(grad_dgz) ) ) ** 2
    bonneval3 =  ( l2_norm ( esperance (l2_norm(grad_dx.T@x))  - esperance(grad_dgz.T@G_z) ) ) ** 2
    val = bonneval1 + lambda1 * bonneval2 + lambda2*bonneval3
    return val


def VGgpt(D, G, E, x, lambda_grad= 0.5):
    z = torch.randn(x.shape[0], 100).to(device)
    grad_real = gradient(D, x)
    grad_fake = gradient(D, G(z))

    term1 = torch.norm(torch.mean(D(x)) - torch.mean(D(G(z))), p=1)
    term2 = torch.norm(torch.mean(grad_real, dim=0) - torch.mean(grad_fake, dim=0), p=2) ** 2

    grad_norm_real = torch.norm(grad_real, p=2)
    grad_norm_fake = torch.norm(grad_fake, p=2)
    grad_penalty = lambda_grad * (torch.mean(grad_norm_real) + torch.mean(grad_norm_fake))

    return term1 + term2 + grad_penalty

    
def gradient(D,x):
    x = x.detach().clone()  
    x.requires_grad = True 
   
    D_output = D(x)
    

    if D_output.dim() > 0:
        # Créer un vecteur de gradients de la même taille que D_output
        grad_outputs = torch.ones_like(D_output)
    else:
        # Si D_output est un scalaire, utiliser un scalaire pour grad_outputs
        grad_outputs = torch.tensor(1.0)
    
    
    D_output.backward(grad_outputs)
    if x.grad is None:
        print('pas bon')
    return x.grad

def afficher_image(X, G,E):
    plt.figure(figsize=(10, 10))
    for i in range(3):
        plt.subplot(5, 5, i+1)
        plt.imshow(X[i].detach().cpu().reshape(28, 28), cmap='gray')
        plt.axis('off')
        
        plt.subplot(5, 5, i+4)
        plt.imshow(G(E(X[i].unsqueeze(0)))[0].detach().cpu().reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.title(f'Reconstruction {i+1}')
    
    plt.tight_layout()
    os.makedirs('Visu_de_encoder', exist_ok=True)
    plt.savefig('Visu_de_encoder/visu.png')
    plt.close()

def afficher_image_random(G, E, device, n_images=3, image_size=(28, 28)):

    plt.figure(figsize=(10, 10))
    
    for i in range(n_images):

        x = torch.randn(1, *image_size).to(device) # Générer une image aléatoire
        
        x_flat = x.view(1, -1)  # vecteur de taille (1, 784)

        G_E_x_flat = G(E(x_flat))  
        G_E_x = G_E_x_flat.view(*image_size).squeeze().detach().cpu()  
        
        x = x.squeeze().detach().cpu()

        # Afficher l'image originale
        plt.subplot(n_images, 2, 2 * i + 1)
        plt.imshow(x, cmap='gray')
        plt.title(f'Image Aléatoire {i+1}')
        plt.axis('off')
        
        # Afficher la reconstruction
        plt.subplot(n_images, 2, 2 * i + 2)
        plt.imshow(G_E_x, cmap='gray')
        plt.title(f'Reconstruction {i+1}')
        plt.axis('off')

    plt.tight_layout()
    os.makedirs('Visu_de_encoder', exist_ok=True)
    plt.savefig('Visu_de_encoder/hasard.png')
    plt.close()



def print_loss(looss_G, looss_D, looss_AE):
    
    os.makedirs('losses/Discriminator', exist_ok=True)
    os.makedirs('losses/Generator', exist_ok=True)
    os.makedirs('losses/Autoencoder', exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(looss_D, label='Discriminator Loss', color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Discriminator Loss')
    plt.legend()
    plt.savefig('losses/Discriminator/discriminator_loss.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(looss_G, label='Generator Loss', color='orange')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Generator Loss')
    plt.legend()
    plt.savefig('losses/Generator/generator_loss.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(looss_AE, label='Autoencoder Loss', color='green')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Autoencoder Loss')
    plt.legend()
    plt.savefig('losses/Autoencoder/autoencoder_loss.png')
    plt.close()

    print("Les courbes de perte ont été sauvegardées.")


def save_models(G, D, AE, folder):
    torch.save(G.state_dict(), os.path.join(folder,'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder,'D.pth'))
    torch.save(AE.state_dict(), os.path.join(folder,'AE.pth'))
    
    
def load_model(G, folder):
    ckpt = torch.load(os.path.join(folder,'G.pth'),map_location=torch.device('cpu'), weights_only=True)
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G

def load_model2(AE, folder, filename):
    
    checkpoint = torch.load(os.path.join(folder, filename), map_location=torch.device('cpu'))
    
    
    new_state_dict = {}
    for k, v in checkpoint.items():
        if 'module.' in k:  # Si les clés contiennent `module.`, les adapter
            new_state_dict[k.replace('module.', '')] = v
        else:  # Sinon, conserver les clés telles quelles
            new_state_dict[k] = v
    
    
    AE.load_state_dict(new_state_dict, strict=False) 
    return AE
