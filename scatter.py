import torch
from itertools import chain
import torchvision
import math
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE



num_epochs = 200
batch_size = 128 
num_cluster = 10
latent_dim = 10

img_transform = transforms.Compose([
    transforms.ToTensor(),
	transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs 
])

dataset_test = MNIST('./data',train=False, transform=img_transform, download=True)
dataloader_test = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=True, drop_last=True)

z_mu_prior = Variable( torch.randn(latent_dim,num_cluster).cuda(), requires_grad=True )
z_var_prior = Variable( torch.ones(latent_dim,num_cluster).cuda() , requires_grad=True )
c_prior = Variable( torch.ones(num_cluster).cuda(), requires_grad=True  )

if not os.path.exists('./img'):
    os.makedirs('./img')

if not os.path.exists('./models'):
    os.makedirs('./models')

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc31 = nn.Linear(256, latent_dim)
        self.fc32 = nn.Linear(256, latent_dim)

        self.fc4 = nn.Linear(latent_dim, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h4 = F.relu(self.fc4(z))
        h5 = F.relu(self.fc5(h4))

        return self.fc6(h5)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar, z


model = VAE()
if torch.cuda.is_available():
    model.cuda()


def get_gamma(z):
    z_t = z.unsqueeze(2).repeat(1,1,num_cluster)
    c_prior_t = c_prior.unsqueeze(0).unsqueeze(1).repeat(batch_size, latent_dim,1)
    z_mu_prior_t = z_mu_prior.unsqueeze(0).repeat(batch_size,1,1)
    z_var_prior_t = z_var_prior.unsqueeze(0).repeat(batch_size,1,1)

    p_c_z = ( torch.sum(  (c_prior_t.log() - 0.5*torch.log(2*math.pi*z_var_prior_t) - (z_t-z_mu_prior_t).pow(2) / 2*(z_var_prior_t)), dim=1)).exp() + 1e-10


    gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)

    return gamma




def VaDE_loss(recon_x, x, z, z_mu, z_logvar):
    # recon_x : [batch_size, size_of_image]
    # x : [batch_size, size_of_image]
    # z : [batch_size, dim]
    # z_mu : [batch_size, dim]
    # z_logvar : [batch_size, dim]
    # z_mu_prior : [batch_size, dim, cluster]
    # z_logvar_prior : [batch_size, dim, cluster]

    z_t = z.unsqueeze(2).repeat(1,1, num_cluster)
    z_mu_t = z_mu.unsqueeze(2).repeat(1,1, num_cluster)
    z_logvar_t = z_logvar.unsqueeze(2).repeat(1,1,num_cluster)

    c_prior_t = c_prior.unsqueeze(0).unsqueeze(0).repeat(batch_size,latent_dim,1)
    z_mu_prior_t = z_mu_prior.unsqueeze(0).repeat(batch_size,1,1)
    z_var_prior_t = z_var_prior.unsqueeze(0).repeat(batch_size,1,1)


    p_c_z = ( torch.sum( c_prior_t.log() - 0.5*torch.log(2*math.pi*z_var_prior_t) - (z_t-z_mu_prior_t).pow(2) / (2*(z_var_prior_t)), dim=1)).exp() + 1e-10

    gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)

    gamma_t = gamma.unsqueeze(1).repeat(1,latent_dim,1)

    recon_loss = F.mse_loss( recon_x, x, reduce=False)
    recon_loss = torch.sum(recon_loss, dim=1)

    kl_pzx_pzc = torch.sum( 0.5*gamma_t* ( latent_dim *( math.log(math.pi*2)) + z_var_prior_t.log()  + z_logvar_t.exp() / z_var_prior_t +\
                (z_mu_t - z_mu_prior_t).pow(2) / z_var_prior_t) , dim=(1,2)) \
                 -0.5*torch.sum( (z_logvar+1), dim=-1) \
                 -torch.sum( (c_prior.unsqueeze(0).repeat(batch_size, 1)).log() * gamma, dim=1)\
                 +torch.sum( gamma.log() * gamma, dim = 1 )
 
        
    loss = torch.mean( kl_pzx_pzc + recon_loss )

    return loss 

map_location = lambda storage, loc: storage
model.load_state_dict(
        torch.load('./models/vae_190.pth', map_location=map_location))


#for batch_idx, data in enumerate(dataloader_test):

data = iter(dataloader_test).next()
print('plot the clusters')
model.eval()
img, label = data
img = img.view(img.size(0), -1)
img = Variable(img)

if torch.cuda.is_available():
    img = img.cuda()

recon_batch, mu, logvar, z = model(img)

mo = TSNE( n_components = 2)
transformed = mo.fit_transform( z.data.cpu().numpy())
z = [ [a,b] for a,b, in zip(label.numpy(), transformed)] 
#s_z = sorted( z, key=lambda e:e[0])

transformed = np.asarray([e[1] for e in z])
labels = np.asarray([e[0] for e in z])

cs = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6','C7', 'C8', 'C9']

for i in range(10000):
    xs = transformed[i,0]
    ys = transformed[i,1]
    l = labels[i]
    plt.scatter(xs, ys, c=cs[l])

plt.savefig('./img/scatter.png')
plt.close()
