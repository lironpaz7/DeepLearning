from __future__ import print_function
import os
import random
import time

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import pandas as pd
import torchvision.utils as vutils
from tqdm import tqdm
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
from IPython.display import HTML
import platform
from torch.utils.data import DataLoader
import utils
import dataset

manualSeed = 42
running_on_linux = 'Linux' in platform.platform()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

size = 64  # real and fake images handled as 3 x size x size
path_to_images = 'img_sample_pt' if not running_on_linux else os.path.join('/home/student/HW3/celebA',
                                                                           f'img_align_celeba_size{size}_pt')
run = "run5"
batch_size = 128  # During training
nc = 3  # Number of channels of RGB
z_ncontinuous = 100  # Size of z continuous latent vector
z_ndiscrete = 20  # Size of z discrete (binary) latent vector
G_nfeatures = 64  # Number of features (filters) for the generator
D_nfeatures = 64  # Number of features (filters) for the discriminator
epochs = 15  # Training epochs
lr = 0.0002  # Learning rate for adam optimizers
netG_path = os.path.join('/home/student/HW3/celebA', f'netG_{run}_discrete{z_ndiscrete}_{epochs}epochs_size{size}')
netD_path = os.path.join('/home/student/HW3/celebA', f'netD_{run}_discrete{z_ndiscrete}_{epochs}epochs_size{size}')


def weights_init(m):
    """ Inspired from the article https://arxiv.org/pdf/1511.06434.pdf"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, z_ncontinuous, z_ndiscrete):
        super(Generator, self).__init__()
        self.z_ncontinuous = z_ncontinuous
        self.z_ndiscrete = z_ndiscrete
        self.de_cnn = nn.Sequential(  # input: z  -> output: fake_images image 3x128x128
            nn.ConvTranspose2d(self.z_ncontinuous + self.z_ndiscrete, G_nfeatures * 8, kernel_size=4, stride=1,
                               padding=0, bias=False),
            nn.BatchNorm2d(G_nfeatures * 8),
            nn.ReLU(True),
            # size: (G_nfeatures*8) x 4 x 4
            nn.ConvTranspose2d(G_nfeatures * 8, G_nfeatures * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(G_nfeatures * 4),
            nn.ReLU(True),
            # size: (G_nfeatures*4) x 8 x 8
            nn.ConvTranspose2d(G_nfeatures * 4, G_nfeatures * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(G_nfeatures * 2),
            nn.ReLU(True),
            # size: (G_nfeatures*2) x 16 x 16
            nn.ConvTranspose2d(G_nfeatures * 2, G_nfeatures, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(G_nfeatures),
            nn.ReLU(True),
            # size: (G_nfeatures) x 32 x 32
            nn.ConvTranspose2d(G_nfeatures, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # size: (nc) x 64 x 64
        )

    def forward(self, input):
        return self.de_cnn(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.cnn = nn.Sequential(  # input: image 3x64x64 -> output: binary decision whether the image is fake_images
            nn.Conv2d(nc, D_nfeatures, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # size: (D_nfeaturesx)32x32
            nn.Conv2d(D_nfeatures, D_nfeatures * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(D_nfeatures * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # size: (D_nfeatures*2)x16x16
            nn.Conv2d(D_nfeatures * 2, D_nfeatures * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(D_nfeatures * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # size: (D_nfeatures*4)x8x8
            nn.Conv2d(D_nfeatures * 4, D_nfeatures * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(D_nfeatures * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # size: (D_nfeatures*8)x4x4
            nn.Conv2d(D_nfeatures * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # size: 1x1x1
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.cnn(input)


def reproduce_hw3():
    print("Fixed Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # RUN ONLY ONCE: download data
    utils.download_data()

    # RUN ONLY ONCE: preprocessing and convert images to tensors
    print('Preprocessing...')
    preprocessing_path = 'img_align_celeba'
    utils.images_preprocessing(size=size, path=preprocessing_path)

    celeb_dataset = dataset.CelebDataset(f'img_align_celeba_size{size}_pt')
    dataloader = DataLoader(celeb_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    print('Preparing for train stage...')
    netG = Generator(z_ncontinuous, z_ndiscrete).to(device)
    netG.apply(weights_init)  # Randomly initialize all weights to mean=0, stdev=0.2.

    netD = Discriminator().to(device)
    netD.apply(weights_init)

    criterion = nn.BCELoss()

    # Create batch of fixed latent vectors to visualize the generator progression
    fixed_continuous_noise = torch.randn(size, z_ncontinuous, 1, 1, device=device)
    fixed_discrete_noise = torch.randint(0, 2, (size, z_ndiscrete, 1, 1), device=device)
    fixed_noise = torch.cat((fixed_continuous_noise, fixed_discrete_noise), dim=1)
    img_list = []  # to visualize the generator progression

    real_label = 1.
    fake_label = 0.
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    G_all_losses = []
    D_all_losses = []
    iters_for_print = 0

    print("Start Training..")
    time.sleep(0.1)
    for epoch in tqdm(range(epochs), desc='Training...', ascii=False, ncols=100):
        for i, data in enumerate(dataloader, 0):
            # Discriminator optimization problem: need to maximize log(D(x)) + log(1 - D(G(z)))
            ## Discriminator: Train with real_images batch ##
            netD.zero_grad()
            real_images = data['images_tensor'].to(device)
            bs = real_images.size(0)
            label = torch.full((bs,), real_label, dtype=torch.float, device=device)

            output = netD(real_images).view(-1)  # Forward pass real_images through Discriminator
            loss_D_real = criterion(output, label)  # Calculate loss on real_images batch
            loss_D_real.backward()
            D_x = output.mean().item()

            ## Discriminator: Train with fake_images batch ##
            # Generate batch of latent vectors
            continuous_noise = torch.randn(bs, z_ncontinuous, 1, 1, device=device)
            discrete_noise = torch.randint(0, 2, (bs, z_ndiscrete, 1, 1), device=device)
            noise = torch.cat((continuous_noise, discrete_noise), dim=1)

            fake_images = netG(noise)  # Generate fake_images image batch with G
            label.fill_(fake_label)
            output = netD(fake_images.detach()).view(-1)  # Forward pass fake_images through Discriminator
            loss_D_fake = criterion(output, label)  # Calculate loss on fake_images batch
            loss_D_fake.backward()
            loss_D = loss_D_real + loss_D_fake  # sum the loss from the real and fake images batches
            optimizerD.step()

            # Generator optimization problem: need to maximize log(D(G(z)))
            netG.zero_grad()
            label.fill_(real_label)  # for the generator cost, fake_images labels are real
            output = netD(fake_images).view(-1)  # Forward pass on fake_images batch through D
            loss_G = criterion(output, label)  # Calculate loss based on the fake images the D catch as fake.
            loss_G.backward()
            D_G_z = output.mean().item()
            optimizerG.step()

            # if i % 200 == 0:
            #     print('[%d/%d][%d/%d]\t,Loss_D: %.4f\tLoss_G: %.4f\t,D(x): %.4f\tD(G(z)): %.4f'
            #           % (epoch + 1, epochs, i, len(dataloader), loss_D.item(), loss_G.item(), D_x, D_G_z))

            G_all_losses.append(loss_G.item())
            D_all_losses.append(loss_D.item())

            # Check the generator progress on fixed_noise
            if (iters_for_print % 500 == 0) or ((epoch == epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake_images = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake_images, padding=2, normalize=True))

            iters_for_print += 1

    pd.DataFrame(
        {'G_loss': G_all_losses,
         'D_Loss': D_all_losses}
    ).to_csv('loss.csv')

    # save model for later use
    torch.save(netG, netG_path)
    torch.save(netD, netD_path)

    # Generator and Discriminator Loss During Training
    # plt.figure(figsize=(10, 5))
    # plt.title("Generator and Discriminator Loss During Training")
    # plt.plot(G_all_losses, label="G")
    # plt.plot(D_all_losses, label="D")
    # plt.xlabel("Batch")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.show()
    #
    # # Visualization of Gâ€™s progression
    # fig = plt.figure(figsize=(8, 8))
    # plt.axis("off")
    # ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    # ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    #
    # html_file = HTML(ani.to_jshtml()).data
    # text_file = open(os.path.join('/home/student/HW3/celebA', "html_output_file.html"), "w")
    # text_file.write(f'{html_file},{run}')
    # text_file.close()
    #
    # # Real Images vs. Fake Images
    # real_batch = next(iter(dataloader))
    # plt.figure(figsize=(15, 15))
    # plt.subplot(1, 2, 1)
    # plt.axis("off")
    # plt.title("Real Images")
    # plt.imshow(np.transpose(vutils.make_grid(real_batch['images_tensor'].to(device)[:64],
    #                                          padding=5, normalize=True).cpu(), (1, 2, 0)))
    # plt.subplot(1, 2, 2)
    # plt.axis("off")
    # plt.title("Fake Images")
    # plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    # plt.show()


if __name__ == "__main__":
    reproduce_hw3()
    netG = torch.load(netG_path).to(device)
    torch.save(netG, 'model_G.pkl')
    netD = torch.load(netD_path).to(device)
    torch.save(netD, 'model_D.pkl')
