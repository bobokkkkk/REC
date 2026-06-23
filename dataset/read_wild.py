import copy
import json
import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import random
from torchvision.transforms import transforms
from tqdm import tqdm

random.seed(42)


class DS(Dataset):
    def __init__(self, datas, transform=None):
        self.data = []
        for data in datas:
            if ('fake_train' in data) or ('fake_test' in data):
                self.data.append((data, 1))
            else:
                self.data.append((data, 0))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label, path


def get_set(path):
    root_path = path
    dicts = next(os.walk(root_path))[1]
    train_data = []
    final_test_data = []
    for d in dicts:
        temp1 = os.path.join(root_path, d)
        ps = next(os.walk(temp1))[1]
        for p in ps:
            temp2 = os.path.join(temp1, p, d[:4])
            pps = next(os.walk(temp2))[1]
            for pp in pps:
                final_path = os.path.join(temp2, pp)
                images = next(os.walk(final_path))[2]
                for image in images:
                    ffinal = os.path.join(final_path, image)
                    if "train" in ffinal:
                        train_data.append(ffinal)
                    else:
                        final_test_data.append(ffinal)

    final_val_data = train_data[:len(final_test_data)]
    final_train_data = train_data[len(final_test_data):]

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    random.shuffle(final_train_data)

    train_set = DS(final_train_data, transform)
    test_set = DS(final_test_data, transform)
    valid_set = DS(final_val_data, transform)
    return train_set, test_set, valid_set


def get_3_loader(path):
    train_set, test_set, valid_set = get_set(path)

    batch_size = 32
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, valid_loader


def get_1_loader(path):
    train_set, test_set, valid_set = get_set(path)

    batch_size = 32
    final_test_set = ConcatDataset([train_set, valid_set, test_set])
    final_test_loader = DataLoader(final_test_set, batch_size=batch_size, shuffle=False)

    return final_test_loader
