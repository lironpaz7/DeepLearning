import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import concurrent.futures
import torchvision.transforms.functional as FT


def resize(image, dims=(224, 224)):
    """
    Resizes the given image to the given dimensions
    :param image: Image object
    :param dims: Dimension's tuple (h,w)
    :return: Resized image object
    """
    return FT.resize(image, dims)

class MasksDataset(Dataset):
    """
    call example: MasksDataset(data_folder=TRAIN_IMG_PATH, split='train')
    """

    def __init__(self, data_folder, split):
        self.split = split.upper()
        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder

        # Read data file names
        self.images = sorted(os.listdir(data_folder))

        # Load data to RAM using multiprocess
        self.loaded_imgs = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.load_single_img, path) for path in self.images]
            self.loaded_imgs = [fut.result() for fut in futures]
        self.loaded_imgs = sorted(self.loaded_imgs, key=lambda x: x[0])  # sort the images to reproduce results
        print(f"Finished loading {self.split} set to memory - total of {len(self.loaded_imgs)} images")

    def __getitem__(self, i):
        # MaskDataset train set
        image_id, image, label = self.loaded_imgs[i]  # str, PIL, tensor

        # Apply transformations and augmentations
        image, label = image.copy(), label.clone()

        # non-fractional for Fast-RCNN
        image = resize(image)  # PIL

        # Convert PIL image to Torch tensor
        image = FT.to_tensor(image)
        return image, label

    def __len__(self):
        return len(self.images)

    def load_single_img(self, path):
        """
        Loads a single image
        :param path: Path to image
        :return:
        """
        image_id, proper_mask = path.strip(".jpg").split("_")
        proper_mask = [1] if proper_mask.lower() == "1" else [0]

        # Read image
        image = Image.open(os.path.join(self.data_folder, path), mode='r').convert('RGB')

        # label
        label = torch.LongTensor(proper_mask)  # (1)

        return image_id, image, label  # str, PIL, tensor
