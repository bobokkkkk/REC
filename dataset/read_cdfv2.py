import os.path
from typing import final

import pandas as pd
from PIL import Image
import random
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from tqdm import tqdm

random.seed(42)

class DS(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = []
        for data in data_path:
            if 'real' in data:
                self.data.append((data, 0))
            if 'synthesis' in data:
                self.data.append((data, 1))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label, path


def process_path(path):
    parts = path.split("\\")
    return "\\".join(parts[4:-1])


def read_excel(path):
    df = pd.read_excel(path, usecols=["path", "describe"])
    df["path"] = df["path"].apply(process_path)
    return df


def get_set(path):
    root_path = path
    dir_list = next(os.walk(root_path))[1]
    test_list_file = next(os.walk(root_path))[2][0]
    total_path = []
    for dir in dir_list:
        path = os.path.join(root_path, dir, 'frames')
        files = next(os.walk(path))[1]
        for file in files:
            fs = next(os.walk(os.path.join(path, file)))[2]
            for f in fs:
                total_path.append(os.path.join(path, file, f))

    with open(os.path.join(root_path, test_list_file), 'rb') as f:
        lines = f.readlines()

    test_list = [item.decode('utf-8').replace('.mp4\n', '').split(' ')[1] for item in lines]
    for i in range(len(test_list)):
        temp = test_list[i].split('/')
        test_list[i] = temp[0] + '\\frames\\' + temp[1]

    test_path = [item for item in total_path if any(sub in item for sub in test_list)]
    total_path = list(set(total_path) - set(test_path))

    random.shuffle(total_path)
    valid_path = total_path[:len(test_path)]
    train_path = total_path[len(test_path):]

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])


    train_set = DS(train_path, transform)
    test_set = DS(test_path, transform)
    valid_set = DS(valid_path, transform)

    return train_set, test_set, valid_set


def get_3_loader(path):
    train_set, test_set, valid_set = get_set(path)

    batch_size = 32
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


def get_1_loader(path):
    train_set, test_set, valid_set = get_set(path)

    batch_size = 32
    final_test_set = ConcatDataset([train_set, valid_set, test_set])
    final_test_loader = DataLoader(final_test_set, batch_size=batch_size, shuffle=False)

    return final_test_loader
