import json
import os.path
import random

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision.transforms import transforms
import numpy as np
import torch


class DS(Dataset):
    def __init__(self, data, transform=None, alternating=False):
        if alternating:
            self.data = self._create_alternating_data(data)
        else:
            self.data = []
            for f in data["fraud"]:
                self.data.append((f, 1))
            for r in data["real"]:
                self.data.append((r, 0))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label, path

    def process_path(self, path):
        parts = path.split("\\")
        return "\\".join(parts[4:-1])

    def _create_alternating_data(self, data):
        fraud_list = [(f, 0) for f in data["fraud"]]
        real_list = [(r, 1) for r in data["real"]]
        min_length = min(len(fraud_list), len(real_list))
        fraud_list = fraud_list[:min_length]
        real_list = real_list[:min_length]

        alternating_data = []
        for i in range(min_length):
            alternating_data.append(real_list[i])
            alternating_data.append(fraud_list[i])

        return alternating_data


def load_json(name):
    with open(name, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def get_data_from_json(name, fraud_data):
    data = []
    for n in name:
        str1 = n[0] + "_" + n[1]
        str2 = n[1] + "_" + n[0]
        for k in fraud_data:
            if (str1 in k) or (str2 in k):
                data.append(k)
    return data


def split_real_data(real_data, train_fraud_len, vaild_fraud_len, test_fraud_len):
    total_fraud = train_fraud_len + vaild_fraud_len + test_fraud_len
    train_ratio = train_fraud_len / total_fraud
    vaild_ratio = vaild_fraud_len / total_fraud
    test_ratio = 1 - train_ratio - vaild_ratio

    train_real_len = int(len(real_data) * train_ratio)
    vaild_real_len = int(len(real_data) * vaild_ratio)
    test_real_len = len(real_data) - train_real_len - vaild_real_len

    train_real = real_data[:train_real_len]
    vaild_real = real_data[train_real_len:train_real_len + vaild_real_len]
    test_real = real_data[train_real_len + vaild_real_len:]

    return train_real, vaild_real, test_real

def get_set(path):
    data_path = path
    train_json = load_json(os.path.join(data_path, "train.json"))
    test_json = load_json(os.path.join(data_path, "test.json"))
    valid_json = load_json(os.path.join(data_path, "val.json"))

    dics = ["Deepfakes", "Face2Face", "NeuralTextures", "FaceSwap"]

    fraud_data = []
    for dic in dics:
        path = os.path.join(data_path, "manipulated_sequences", dic, "c23", "frames")
        for root, _, files in os.walk(path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    fraud_data.append(os.path.join(root, file))

    train_data = get_data_from_json(train_json, fraud_data)
    test_data = get_data_from_json(test_json, fraud_data)
    valid_data = get_data_from_json(valid_json, fraud_data)

    real_data = []
    real_root = os.path.join(data_path, 'original_sequences', 'youtube', 'c23', 'frames')
    for root, _, files in os.walk(real_root):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                real_data.append(os.path.join(root, file))

    train_real, valid_real, test_real = split_real_data(real_data, len(train_data), len(valid_data), len(test_data))

    train_data = random.sample(train_data, len(train_real))
    valid_data = random.sample(valid_data, len(valid_real))
    test_data = random.sample(test_data, len(test_real))

    final_train = {"fraud": train_data, "real": train_real}
    final_test = {"fraud": test_data, "real": test_real}
    final_valid = {"fraud": valid_data, "real": valid_real}

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    train_set = DS(final_train, transform)
    test_set = DS(final_test, transform)
    valid_set = DS(final_valid, transform)

    return train_set, test_set, valid_set


def get_3_loader(path):
    train_set, test_set, valid_set = get_set(path)

    batch_size = 32
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


def get_2_loader(path):
    train_set, test_set, valid_set = get_set(path)

    final_train_set = ConcatDataset([train_set, valid_set])

    batch_size = 32
    train_loader = DataLoader(final_train_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
