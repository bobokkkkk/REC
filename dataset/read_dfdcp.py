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
            if data['label'] == 'fake':
                self.data.append((data['path'], 1))
            else:
                self.data.append((data['path'], 0))
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


def get_set(path):
    root_path = path
    dicts = next(os.walk(root_path))[1]
    split_file = next(os.walk(root_path))[2][0]
    split_path = os.path.join(root_path, split_file)
    with open(split_path, 'r') as f:
        split_datas = json.load(f)

    train_data = []
    final_test_data = []

    for data in split_datas:
        split_datas[data]['path'] = os.path.join(root_path, str(data).split('/')[0], 'frames',
                                                 str(data).split('/')[-1].replace('.mp4', ''))
        if os.path.exists(split_datas[data]['path']):
            if split_datas[data]['set'] == 'train':
                pics = next(os.walk(split_datas[data]['path']))[2]
                for pic in pics:
                    temp = copy.deepcopy(split_datas[data])
                    temp['path'] = os.path.join(split_datas[data]['path'], pic)
                    train_data.append(temp)
            else:
                pics = next(os.walk(split_datas[data]['path']))[2]
                for pic in pics:
                    temp = copy.deepcopy(split_datas[data])
                    temp['path'] = os.path.join(split_datas[data]['path'], pic)
                    final_test_data.append(temp)

    random.shuffle(train_data)

    final_val_data = train_data[:len(final_test_data)]
    final_train_data = train_data[len(final_test_data):]

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

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
