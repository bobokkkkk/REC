import json
import os.path
import random

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.transforms import transforms
from tqdm import tqdm

random.seed(42)


class DS(Dataset):
    def __init__(self, data, transform=None):
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


def load_json(name):
    with open(name, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def get_data_from_excel(name, fraud_data):
    data = set()
    for n in name:
        temp = n.split("/")
        img_folder = temp[4].split('_', 1)[1].rsplit('_', 1)[0]
        for k in fraud_data:
            if img_folder in k:
                data.add(k)
    return list(data)


def split_real_data(real_data, train_fraud_len, valid_fraud_len, test_fraud_len):
    total_fraud = train_fraud_len + valid_fraud_len + test_fraud_len

    train_ratio = train_fraud_len / total_fraud
    valid_ratio = valid_fraud_len / total_fraud
    test_ratio = 1 - train_ratio - valid_ratio

    random.shuffle(real_data)

    train_real_len = int(len(real_data) * train_ratio)
    valid_real_len = int(len(real_data) * valid_ratio)
    test_real_len = len(real_data) - train_real_len - valid_real_len

    train_real = real_data[:train_real_len]
    valid_real = real_data[train_real_len:train_real_len + valid_real_len]
    test_real = real_data[train_real_len + valid_real_len:]

    return train_real, valid_real, test_real



def read_data_from_excel(path):
    df = pd.read_excel(path, usecols=["img_path"])
    return df

def get_set(path):
    data_path = path
    test_path = read_data_from_excel(os.path.join(path, "test.xlsx"))['img_path'].tolist()

    dics = ["DeepFakeDetection"]

    fraud_data = []
    for dic in dics:
        path = os.path.join(data_path, "manipulated_sequences", dic, "c23", "frames")
        for root, _, files in os.walk(path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    fraud_data.append(os.path.join(root, file))

    test_data = get_data_from_excel(test_path, fraud_data)
    rest_data = [x for x in fraud_data if x not in test_data]
    train_data = rest_data[len(test_data):]
    valid_data = rest_data[:len(test_data)]

    real_data = []
    real_root = os.path.join(data_path, 'original_sequences', 'actors', 'c23', 'frames')
    for root, _, files in os.walk(real_root):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                real_data.append(os.path.join(root, file))

    test_real = get_data_from_excel(test_path, real_data)
    rest_data = [x for x in real_data if x not in test_real]
    train_real = rest_data[len(test_real):]
    valid_real = rest_data[:len(test_real)]


    train_data = random.sample(train_data, len(train_real))

    final_train = {"fraud": train_data, "real": train_real}
    final_test = {"fraud": test_data, "real": test_real}
    final_valid = {"fraud": valid_data, "real": valid_real}

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_set = DS(final_train, transform=transform)
    test_set = DS(final_test, transform=transform)
    valid_set = DS(final_valid, transform=transform)

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

    final_test_set = ConcatDataset([train_set, valid_set, test_set])

    batch_size = 32
    test_loader = DataLoader(final_test_set, batch_size=batch_size, shuffle=False)

    return test_loader
