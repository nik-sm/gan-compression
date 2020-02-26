import pathlib
import params as P
import torch.utils.data 

AVAIL_DATALOADERS = ['celeba','ffhq']

def celeba(train=True):
    if train:
        path = './data/celeba-preprocessed/train'
    else:
        path = './data/celeba-preprocessed/test'
    return torch.utils.data.DataLoader(
            FolderDataset(path),
            batch_size=P.batch_size,
            shuffle=False,
            num_workers=2)

def ffhq(train=True):
    if train:
        path = './data/ffhq-preprocessed/train'
    else:
        path = './data/ffhq-preprocessed/test'
    return torch.utils.data.DataLoader(
            FolderDataset(path),
            batch_size=P.batch_size,
            shuffle=False,
            num_workers=2)

class FolderDataset(torch.utils.data.Dataset):
    """
    Simplified version of ImageFolder that does not require subfolders.
    - Used for loading images only (without any labels).
    - Assumes images are pre-processed into tensors and saved as '*.pt' for optimal loading speed.
    """
    def __init__(self, folder):
        self.images = sorted(pathlib.Path(folder).rglob('*.pt'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return torch.load(self.images[i])
