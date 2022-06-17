import os
import platform
from torch.utils.data import Dataset
import utils
from torch.utils.data import DataLoader
import main


class CelebDataset(Dataset):
    """celebA dataset class"""

    def __init__(self, images_path: str):
        self.images = utils.load_images(images_path)  # all images as dictionary {'000001' : torch.tensor ...}

    def __getitem__(self, idx):
        image_id = str(idx + 1).zfill(6)
        return {'images_id': image_id, 'images_tensor': self.images[image_id]}

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    size = main.size
    running_on_linux = 'Linux' in platform.platform()
    path = 'img_sample_pt' if not running_on_linux else os.path.join('/home/student/HW3/celebA',
                                                                     'img_align_celeba' + f'_size{size}_pt')
    celeb_dataset = CelebDataset(path)

    batch_size = 6
    dataloader = DataLoader(celeb_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    batch = next(iter(dataloader))
