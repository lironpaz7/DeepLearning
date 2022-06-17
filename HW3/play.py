from __future__ import print_function
from main import Generator, Discriminator
import main
import os
import random
import torch.nn.parallel
import torch.utils.data
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import os
import platform
import utils
import reverse_generator
import matplotlib.animation as animation
from IPython.display import HTML


def get_attributes_file(path):
    attr_dict = dict()  # {image_id: -1/1 vector for 41 attributes}
    file = open(path, "r")
    for idx, line in enumerate(file):
        if idx == 0:
            continue
        elif idx == 1:
            header = line.split()
        else:
            attr_dict[line[:6]] = line.split()[1:]

    return attr_dict, header


def plot_images_for_animation(title, images, nrow=8):
    return plt.imshow(np.transpose(vutils.make_grid(images.cpu().detach(), padding=2, normalize=True), (1, 2, 0)),
                      animated=True)


def plot_images(title, images, nrow=8):
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title(title)
    plt.imshow(np.transpose(vutils.make_grid(images.cpu().detach()[:64], nrow=nrow,
                                             padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()


def get_difference_vector_between_groups(A_name, B_name, latent_vectors_A, latent_vectors_B):
    """
    :param A_name: representative name for group A
    :param B_name: representative name for group B
    :param A_idx: list of indices of group A members, in the latent vectors tensors
    :param B_idx: list of indices of group B members, in the latent vectors tensors
    :param latent_vectors: aka fixed_noise, to reproduce the same fake_images images
    :return: difference_vector: group_B - group_A
    """
    A_images = netG(latent_vectors_A)
    B_images = netG(latent_vectors_B)

    # plot_images(f'both groups fake_images images', fake_images)
    # plot_images(f'{A_name} fake_images images', A_images)
    # plot_images(f'{B_name} fake_images images', B_images)

    A_average_vector = torch.mean(latent_vectors_A, dim=0)
    B_average_vector = torch.mean(latent_vectors_B, dim=0)
    difference_vector = B_average_vector - A_average_vector

    A_to_B_images = netG((latent_vectors_A + difference_vector).reshape([-1, latent_vectors_A.shape[1], 1, 1]))
    plot_images(f'{A_name} and {A_name}_to_{B_name} fake_images images', torch.cat((A_images, A_to_B_images),
                                                                                   dim=0), len(A_images))

    B_to_A_images = netG((latent_vectors_B - difference_vector).reshape([-1, latent_vectors_B.shape[1], 1, 1]))
    plot_images(f'{B_name} and {B_name}_to_{A_name} fake_images images', torch.cat((B_images, B_to_A_images),
                                                                                   dim=0), len(B_images))
    return difference_vector


def get_difference_vector_between_groups_idx(A_name, B_name, A_idx, B_idx, latent_vectors):
    """
    :param A_name: representative name for group A
    :param B_name: representative name for group B
    :param A_idx: list of indices of group A members, in the latent vectors tensors
    :param B_idx: list of indices of group B members, in the latent vectors tensors
    :param latent_vectors: aka fixed_noise, to reproduce the same fake_images images
    :return: difference_vector: group_B - group_A
    """
    fake_images = netG(latent_vectors)
    A_images = fake_images[A_idx]
    B_images = fake_images[B_idx]

    # plot_images(f'both groups fake_images images', fake_images)
    plot_images(f'{A_name} fake_images images', A_images)
    plot_images(f'{B_name} fake_images images', B_images)

    A_average_vector = torch.mean(latent_vectors[A_idx], dim=0)
    B_average_vector = torch.mean(latent_vectors[B_idx], dim=0)
    difference_vector = B_average_vector - A_average_vector

    A_to_B_images = netG((latent_vectors[A_idx] + difference_vector).reshape([-1, latent_vectors.shape[1], 1, 1]))
    plot_images(f'{A_name} and {A_name}_to_{B_name} fake_images images', torch.cat((A_images, A_to_B_images),
                                                                                   dim=0), len(A_images))

    B_to_A_images = netG((latent_vectors[B_idx] - difference_vector).reshape([-1, latent_vectors.shape[1], 1, 1]))
    plot_images(f'{B_name} and {B_name}_to_{A_name} fake_images images', torch.cat((B_images, B_to_A_images),
                                                                                   dim=0), len(B_images))
    return difference_vector


def get_original_images_by_attribute(attribute, size):
    attr_path = '/datashare/list_attr_celeba.txt'
    attr_dict, header = get_attributes_file(attr_path)

    attr_df = pd.DataFrame.from_dict(attr_dict, orient='index', columns=header)
    for col in attr_df.columns:
        attr_df[col] = pd.to_numeric(attr_df[col])

    images_id_group_A = attr_df.index[attr_df[attribute] == 1].tolist()
    random.shuffle(images_id_group_A)
    images_id_group_A = images_id_group_A[:6]
    images_id_group_B = attr_df.index[attr_df[attribute] == -1].tolist()
    random.shuffle(images_id_group_B)
    images_id_group_B = images_id_group_B[:6]

    dataroot = 'img_sample_pt' if not running_on_linux else os.path.join('/home/student/HW3/celebA',
                                                                         f'img_align_celeba')

    files_names = [os.path.join(dataroot + f'_size{size}_pt', image_id + ".pt") for image_id in images_id_group_A]
    A_tensors = torch.stack([torch.load(f) for f in files_names])

    files_names = [os.path.join(dataroot + f'_size{size}_pt', image_id + ".pt") for image_id in images_id_group_B]
    B_tensors = torch.stack([torch.load(f) for f in files_names])

    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title(f'{attribute} Images')
    plt.imshow(np.transpose(vutils.make_grid(A_tensors.to(device)[:64], nrow=2,
                                             padding=5, normalize=True).cpu(), (1, 2, 0)))

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title(f'non {attribute} Images')
    plt.imshow(np.transpose(vutils.make_grid(B_tensors.to(device)[:64], nrow=2,
                                             padding=5, normalize=True).cpu(), (1, 2, 0)))
    plt.show()
    return A_tensors.to(device), B_tensors.to(device)


def plot_images_with_approx_z(niter, images):
    z_approx_A = reverse_generator.reverse_generator(G=netG, images=images, niter=niter)
    generated_images_A = netG(z_approx_A)

    plt.figure(figsize=(6, 6))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(
        np.transpose(vutils.make_grid(images.to(device)[:64], nrow=2, padding=5, normalize=True).cpu(), (1, 2, 0)))
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title(f'Generated Images, {niter} iters_for_print')
    plt.imshow(
        np.transpose(vutils.make_grid(generated_images_A.cpu().detach()[:64], nrow=2, padding=5, normalize=True).cpu(),
                     (1, 2, 0)))
    plt.show()
    return z_approx_A


def create_gif_discrete_values_change():
    """Visualization of discrete values change. saves the gif to html_Visualization_discrete_values_change.html"""
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    images = list()
    col_idx = list()
    fixed_continuous_noise = torch.randn(images_amount, z_ncontinuous, 1, 1, device=device)

    for i in range(z_ndiscrete):
        col_idx.append(i)
        fixed_discrete_noise = torch.zeros(images_amount, z_ndiscrete, 1, 1, device=device)
        fixed_discrete_noise[torch.arange(0, images_amount), torch.tensor(col_idx).reshape(-1, 1)] = True
        fixed_noise = torch.cat((fixed_continuous_noise, fixed_discrete_noise), dim=1)
        fixed_noise_images = netG(fixed_noise.reshape([-1, fixed_noise.shape[1], 1, 1]))

        images.append([plot_images_for_animation(f'fixed_noise images', fixed_noise_images)])

    ani = animation.ArtistAnimation(fig, images, interval=1000, repeat_delay=1000, blit=True)
    html_file = HTML(ani.to_jshtml()).data
    text_file = open(os.path.join('/home/student/HW3/celebA', "html_Visualization_discrete_values_change.html"), "w")
    text_file.write(f'{html_file}')
    text_file.close()


def get_difference_vector_between_approx_z():
    """Infer the approx. z of labeled real image, and find the difference that capture an attribute"""
    # Try to get difference vector of attributes ##
    attributes = ['Bald', 'Big_Lips', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Chubby', 'Eyeglasses',
                  'Gray_Hair', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Smiling',
                  'Straight_Hair', 'Wavy_Hair', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necktie', 'Young']
    niter = 20000
    for attribute in attributes:
        non_attribute = "Non_" + attribute
        A_images, B_images = get_original_images_by_attribute(attribute, size)
        z_approx_A = plot_images_with_approx_z(niter, A_images)
        z_approx_B = plot_images_with_approx_z(niter, B_images)

        get_difference_vector_between_groups(attribute, non_attribute, z_approx_A, z_approx_B)


if __name__ == "__main__":
    running_on_linux = 'Linux' in platform.platform()
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    manualSeed = 10  # run.manualSeed
    print("Fixed Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    netG = torch.load(main.netG_path).to(device)
    netD = torch.load(main.netD_path).to(device)
    z_ncontinuous = main.z_ncontinuous
    z_ndiscrete = main.z_ndiscrete
    images_amount = 64
    size = main.size

    fixed_continuous_noise = torch.randn(images_amount, main.z_ncontinuous, 1, 1, device=device)
    fixed_discrete_noise = torch.randint(0, 2, (images_amount, z_ndiscrete, 1, 1), device=device)
    fixed_noise = torch.cat((fixed_continuous_noise, fixed_discrete_noise), dim=1)
    fixed_noise_images = netG(fixed_noise.reshape([-1, fixed_noise.shape[1], 1, 1]))
    plot_images("fake images", fixed_noise_images)

    # try approx z plots
    # get_difference_vector_between_approx_z()

    # for run2_30epochs, manualSeed = 999
    # men = [8, 9, 21, 23, 24, 30, 33, 37, 38, 41]
    # women = [1, 4, 5, 12, 14, 19, 25, 27, 28, 44, 59, 61]
    # get_difference_vector_between_groups("men", "women", fixed_noise[men], fixed_noise[women])

    # for netG_run5_discrete20_15epochs_size64, manualSeed=42
    # blond_hair = [4,11,35,36,37,42,45]
    # black_hair = [25,26,29,47,54]
    # get_difference_vector_between_groups_idx("blond_hair", "black_hair", blond_hair, black_hair, fixed_noise)

    # for netG_run5_discrete20_15epochs_size64, manualSeed=42
    # smiling = [2,20,22,24,31,49,59,62]
    # not_smiling = [5,7,17,26,29,32,40,53]
    # get_difference_vector_between_groups_idx("smiling", "not_smiling", smiling, not_smiling, fixed_noise)
