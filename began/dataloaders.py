import os
import numpy as np
import pathlib
import params as P
import torch.utils.data 

AVAIL_DATALOADERS = ['celeba','ffhq']

def get_dataloader(base_path, n_train=-1, train=True):
    if train:
        path = os.path.join(base_path, 'train')
    else:
        path = os.path.join(base_path, 'test')

    if n_train > 0:
        dataset = torch.utils.data.Subset(
            FolderDataset(path),
            np.arange(n_train))
    else:
        dataset = FolderDataset(path)

    return torch.utils.data.DataLoader(
            dataset,
            batch_size=P.batch_size,
            shuffle=False,
            num_workers=0)

class FolderDataset(torch.utils.data.Dataset):
    """
    Simplified version of ImageFolder that does not require subfolders.
    - Used for loading images only (without any labels).
    - Assumes images are pre-processed into tensors and saved as '*.pt' for optimal loading speed.
    """
    def __init__(self, folder):
        self.folder = folder
        self.images = sorted(pathlib.Path(folder).rglob('*.pt'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return torch.load(self.images[i])

    def __repr__(self):
        return self.__class__ + ":\n" + \
               f"Images folder: {self.folder}" + \
               f"Number of images: {self.__len__}"
