import argparse
from pathlib import Path
import os
from os.path import exists, join, isdir, split, splitext
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import torch

import params as P

def process(input_folder, output_folder, transform, glob='*.png', test_fraction=0.05):
    """
    This processing assumes the input images, when sorted by name, are randomized.
    - We produce a single folder of 'train' and 'test' data.
    - The first consecutive chunk is used for train, and the next for test.
    """
    finished_flag = join(output_folder, 'PREPROCESSING_COMPLETED.flag')
    if exists(finished_flag):
        print(f"Already preprocessed (remove {finished_flag} file to rerun)")
        return

    print(f"Processing: {input_folder} to {output_folder}")

    all_image_files = sorted(Path(input_folder).rglob(glob))

    n_total = len(all_image_files)
    # Keep at least 1 for testing from each folder
    n_test = max(1, int(n_total * test_fraction)) 
    n_train = n_total - n_test
    train_files = all_image_files[:n_train]
    test_files = all_image_files[n_train:]
    print(f"Total number of images: {n_total}")
    print(f"Number for train: {n_train}")
    print(f"Number for test: {n_test}")

    # Save train subset
    os.makedirs(join(output_folder, 'train'), exist_ok=True)
    for img_full_path in tqdm(train_files, leave=True, desc='train_subset'):
        _, filename = split(img_full_path)
        img_name, ext = splitext(filename)
        img = transform(Image.open(img_full_path))
        destination = join(output_folder, 'train', img_name + '.pt')
        #print(f"Would save: {img.shape} to {destination}")
        torch.save(img, destination)
 
    # Save test subset
    os.makedirs(join(output_folder, 'test'), exist_ok=True)
    for img_full_path in tqdm(test_files, leave=True, desc='test_subset'):
        _, filename = split(img_full_path)
        img_name, ext = splitext(filename)
        img = transform(Image.open(img_full_path))
        destination = join(output_folder, 'test', img_name + '.pt')
        #print(f"Would save: {img.shape} to {destination}")
        torch.save(img, destination)

    open(finished_flag, 'a').close()
    return

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['ffhq','celeba'], required=True)
    a = p.parse_args()

    transform = transforms.Compose([
        transforms.Resize((P.size, P.size)),
        transforms.ToTensor()
        ])

    if a.dataset == 'ffhq':
        input_folder = '/data/niklas/ffhq-dataset/images1024x1024/'
        output_folder = './data/ffhq-preprocessed'
        glob = '*.png'
    elif a.dataset == 'celeba':
        input_folder = '/data/niklas/datasets/celeba/'
        output_folder = './data/celeba-preprocessed'
        glob = '*.jpg'
    process(input_folder, output_folder, transform, glob)
