import torch
import os
import platform
from PIL import Image
from torchvision import transforms
import concurrent.futures

from tqdm import tqdm

import main


def download_data():
    """
        downloads the data and unzips, delete zip when finish
    """
    print("Downloading celeba.zip ...")
    os.system('wget https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip')
    print("Unzipping celeba.zip ...")
    os.system('unzip celeba.zip')
    print("Deleting file celeba.zip ...")
    os.system('rm celeba.zip')


def images_preprocessing(size, path):
    """
    :param size: resize images to 3*size*size
    :param path: path to images folder
    """
    if not os.path.exists(path + f'_size{size}_pt'):
        os.makedirs(path + f'_size{size}_pt')

    files_names = os.listdir(path)
    transform = transforms.Compose([transforms.Resize(size=(size, size)),
                                    transforms.ToTensor(),  # move to tensor and normalize to [0,1]
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # normalize to [-1,1]

    for file_name in tqdm(files_names, desc='Processing...'):
        img_path = os.path.join(path, file_name)
        image = Image.open(img_path).convert('RGB')
        image = transform(image)
        save_path = os.path.join(path + f'_size{size}_pt', file_name.split('.')[0] + '.pt')
        torch.save(image, save_path)  # if we wish float16 >>> torch.save(image.to(dtype=torch.float16), save_path))


def read_one_tensor(fname, path):
    """
        read one .pt tensor from disk
    :param fname: filename like '000001.pt'
    :param path: path for where fname is located
    :return: the string of the file name without extension and the image tensor
    """
    image = torch.load(os.path.join(path, fname))
    return fname.split('.')[0], image


def load_images(path):
    """
        load all images from path to dictionary with preprocessing
    :param path: path to .pt images
    :return: all images as dictionary {'000001' : torch.tensor ...}
    """
    files_names = sorted(os.listdir(path))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(read_one_tensor, f, path) for f in files_names]
        return {img_id: img_tensor for img_id, img_tensor in [fut.result() for fut in futures]}


if __name__ == '__main__':
    # original images size == (178, 218)
    running_on_linux = 'Linux' in platform.platform()
    size = main.size
    path = 'img_sample' if not running_on_linux else f'/home/student/HW3/celebA/img_align_celeba'
    # images_preprocessing(size=size, path=path)
    # load_images(path + '_pt')
