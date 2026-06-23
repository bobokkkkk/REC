import os
from pathlib import Path
import argparse
import yaml
import torch
from torch.optim import AdamW
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, confusion_matrix
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from dataset import read_dfdcp, read_dfd, read_wild, read_cdfv2
from dataset.read_ff import get_3_loader as ffpp_get_3_loader
from modules.model import RECC
from noise_utils import add_noise_to_tensor

e_data = []

def get_extra_data_path(path):
    pics = next(os.walk(path))[2]
    files = next(os.walk(path))[1]
    if len(pics) != 0:
        for i in range(len(pics)):
            e_data.append(os.path.join(path, pics[i]))
    else:
        for i in range(len(files)):
            get_extra_data_path(os.path.join(path, files[i]))

def crop_string(string):
    return "\\".join(string.split("\\")[5:-1])

def get_img(path, transform):
    same_path = []
    for i in range(len(path)):
        same_path.append("\\".join(path[i].split("\\")[4:-1]))
    common_items = []
    common_img = []
    for target_path in same_path:
        for item in e_data:
            crop_item = crop_string(item)
            if crop_item == target_path:
                common_items.append(item)
                img = Image.open(item).convert("RGB")
                if transform is not None:
                    img = transform(img)
                common_img.append(img)
                break
    return common_img

def apply_tensor_noise(images, noise_prob=0.3, noise_level=0.1):
    if torch.rand(1).item() < noise_prob:
        noise_types = ['gaussian', 'uniform']
        noise_type = np.random.choice(noise_types)
        return add_noise_to_tensor(images, noise_type, noise_level)
    return images

def train(model, train_loader, valid_loader, optimizer, epochs, transform=None, save_path="log/commd.pth"):
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.6, patience=4, min_lr=5e-8
    )

    best_auc = 0.0
    Path(Path(save_path).parent).mkdir(parents=True, exist_ok=True)
    early_stop_counter = 0
    cross_loss = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler(
        init_scale=2 ** 10, growth_factor=1.5, backoff_factor=0.8, growth_interval=1000
    )

    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0

        epoch_labels = []
        epoch_probs = []
        epoch_pred = []

        progress1 = tqdm(train_loader, f"{epoch + 1}/{epochs}")
        for image, label, path in progress1:
            image = image.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            e_img = get_img(path, transform)
            e_img = torch.stack(e_img, dim=0).cuda(non_blocking=True)

            output = model(image, e_img, label)

            feature = output["feature"]
            e_feature = output["e_feature"]
            rcon_feature = output["rcon_feature"]
            crop_feature = output["crop_features"]
            recon_crop_feature = output["recon_crop_feature"]

            pred = output["predict"].argmax(dim=1)
            probs = torch.softmax(output["predict"], dim=1)[:, 1]

            epoch_labels.extend(label.detach().cpu().numpy())
            epoch_probs.extend(probs.detach().cpu().numpy())
            epoch_pred.extend(pred.detach().cpu().numpy())
            auc = roc_auc_score(epoch_labels, epoch_probs)
            correct += (pred == label).sum().item()
            total += label.size(0)

            classification_loss = cross_loss(output["predict"], label)
            classification_loss = abs(classification_loss - 0.015) + 0.015

            feature_pooled = torch.mean(feature, dim=1)
            rcon_feature_pooled = torch.mean(rcon_feature, dim=1)
            e_feature_pooled = torch.mean(e_feature, dim=1)
            crop_feature_pooled = torch.mean(crop_feature, dim=1)
            recon_crop_feature_pooled = torch.mean(recon_crop_feature, dim=1)

            fake_real_contrastive_loss = 0.0
            fake_e_contrastive_loss = 0.0
            recon_loss = 0.0
            real_e_pull_loss = 0.0
            fine_grad_loss = 0.0

            real_mask = (label == 0)
            fake_mask = (label == 1)

            if real_mask.any() and fake_mask.any():
                real_features = feature_pooled[real_mask]
                fake_features = feature_pooled[fake_mask]
                fake_real_sims = []
                for fake_feat in fake_features:
                    sims = F.cosine_similarity(
                        fake_feat.unsqueeze(0).expand(real_features.size(0), -1),
                        real_features, dim=1
                    )
                    fake_real_sims.append(sims.mean())
                fake_real_similarities = torch.stack(fake_real_sims)
                margin = 0.15
                fake_real_contrastive_loss = torch.clamp(fake_real_similarities - margin, min=0).mean()

            if fake_mask.any() and e_feature_pooled is not None:
                fake_features = feature_pooled[fake_mask]
                e_features = e_feature_pooled[fake_mask]
                fake_e_similarities = F.cosine_similarity(fake_features, e_features, dim=1)
                margin_e = 0.08
                fake_e_contrastive_loss = torch.clamp(fake_e_similarities - margin_e, min=0).mean()

            if real_mask.any():
                real_original = feature_pooled[real_mask]
                real_reconstructed = rcon_feature_pooled[real_mask]
                recon_loss = F.mse_loss(real_reconstructed, real_original)
                fine_grad_loss = F.mse_loss(recon_crop_feature_pooled, crop_feature_pooled)

            if fake_mask.any():
                fake_original = feature_pooled[fake_mask]
                fake_reconstructed = rcon_feature_pooled[fake_mask]
                recon_loss = recon_loss + F.mse_loss(fake_reconstructed, fake_original)

            if real_mask.any() and e_feature_pooled is not None:
                real_features = feature_pooled[real_mask]
                real_e_features = e_feature_pooled[real_mask]
                real_e_similarities = F.cosine_similarity(real_features, real_e_features, dim=1)
                real_e_pull_loss = (1.0 - real_e_similarities).mean()

            total_loss = (classification_loss +
                          0.008 * fake_real_contrastive_loss +
                          0.004 * fake_e_contrastive_loss +
                          0.4 * recon_loss +
                          0.003 * real_e_pull_loss +
                          0.4 * fine_grad_loss)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            progress1.set_postfix({
                "cls_loss": float(classification_loss.item()),
                "fake_real_contra": float(fake_real_contrastive_loss if isinstance(fake_real_contrastive_loss, float) else fake_real_contrastive_loss.item()),
                "fake_e_contra": float(fake_e_contrastive_loss if isinstance(fake_e_contrastive_loss, float) else fake_e_contrastive_loss.item()),
                "recon_loss": float(recon_loss if isinstance(recon_loss, float) else recon_loss.item()),
                "real_e_pull": float(real_e_pull_loss if isinstance(real_e_pull_loss, float) else real_e_pull_loss.item()),
                "fine_grad": float(fine_grad_loss if isinstance(fine_grad_loss, float) else fine_grad_loss.item()),
                "acc": correct / max(1, total),
                "auc": auc
            })

        matrix = confusion_matrix(epoch_labels, epoch_pred)
        print(matrix)

        acc, auc = validate(model, valid_loader)
        scheduler.step(auc)
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), save_path)
            print(f"Model saved! Best AUC: {best_auc:.4f}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"Early stop counting: {early_stop_counter}.")
            if early_stop_counter == 2:
                print("Early stop.")
                break

def validate(model, valid_loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    model.flag = True

    progress = tqdm(valid_loader, desc="validate", unit="batch")
    with torch.no_grad():
        for images, labels, path in progress:
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            logits = model(images, label=labels, e_img=None)
            pred = logits["predict"].argmax(dim=1)
            probs = torch.softmax(logits["predict"], dim=1)[:, 1]

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            acc = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()
            auc = roc_auc_score(all_labels, all_probs)
            progress.set_postfix(acc=f"{acc:.4f}", auc=f"{auc:.4f}")

    auc = roc_auc_score(all_labels, all_probs)
    print(f'\nAUC: {auc}\nACC: {acc}')
    model.flag = False
    return acc, auc

def test(model, test_loader, save_path):
    print("Loading best model...")
    model.load_state_dict(torch.load(save_path, map_location="cuda"))
    model.eval()
    model.flag = True
    all_preds, all_labels, all_probs = [], [], []

    progress = tqdm(test_loader, desc="test", unit="batch")
    with torch.no_grad():
        for images, labels, path in progress:
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            logits = model(images, label=labels, e_img=None)
            pred = logits["predict"].argmax(dim=1)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            probs = torch.softmax(logits["predict"], dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            acc = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()
            progress.set_postfix(acc=f"{acc:.4f}")

    auc = roc_auc_score(all_labels, all_probs)
    print(f"\nAUC Score: {auc:.4f}\n")
    return acc, auc

def build_loaders(trainset_key, valset_key, testset_key, paths, batch_size, num_workers, ff_kwargs=None):
    registry = {
        "ffpp":  (ffpp_get_3_loader, paths.get("ffpp")),
        "cdfv2": (read_cdfv2.get_3_loader, paths.get("cdfv2")),
        "dfdcp": (read_dfdcp.get_3_loader, paths.get("dfdcp")),
        "dfd":   (read_dfd.get_3_loader, paths.get("dfd")),
        "wild":  (read_wild.get_3_loader, paths.get("wild")),
    }

    def _get(key):
        if key is None:
            return None, None, None
        fn, root = registry[key]
        if root is None:
            raise ValueError(f"Path for dataset '{key}' not provided in config.")
        return fn(root)

    train_loaders = _get(trainset_key)
    val_loaders = _get(valset_key)
    test_loaders = _get(testset_key)

    train_loader = train_loaders[0] if train_loaders else None
    val_loader   = val_loaders[1]   if val_loaders else None
    test_loader  = test_loaders[2]  if test_loaders else None
    return train_loader, val_loader, test_loader

def parse_args():
    p = argparse.ArgumentParser(description="REC trainer")
    p.add_argument('-config', '--config', type=str, default='configs/rec.yaml', help='YAML config file')
    p.add_argument('-trainset', '--trainset', type=str, default='ffpp', choices=['ffpp','cdfv2','dfdcp','dfd','wild'], help='Train dataset key')
    p.add_argument('-valset', '--valset', type=str, default='cdfv2', choices=['ffpp','cdfv2','dfdcp','dfd','wild'], help='Validation dataset key')
    p.add_argument('-testset', '--testset', type=str, default='cdfv2', choices=['ffpp','cdfv2','dfdcp','dfd','wild'], help='Test dataset key')
    p.add_argument('-epochs', '--epochs', type=int, default=30)
    p.add_argument('-save_path', '--save_path', type=str, default='log/best_model_with_train_all.pth')
    p.add_argument('-device', '--device', type=str, default='cuda')
    p.add_argument('-seed', '--seed', type=int, default=42)
    p.add_argument('-extra_data', '--extra_data', type=str, default=None, help='Path to extra/crop images for e_data')
    p.add_argument('-test_only', '--test_only', action='store_true', help='Skip training, only run test using --save_path')
    return p.parse_args()

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def build_optimizer(model, cfg_optim):
    param_groups = [
        {'params': model.clip_model.parameters(),
         'lr': cfg_optim['clip_model']['lr'],
         'weight_decay': cfg_optim['clip_model']['weight_decay']},
        {'params': model.Unet1.parameters(),
         'lr': cfg_optim['unet1']['lr'],
         'weight_decay': cfg_optim['unet1']['weight_decay']},
        {'params': model.Classifier.parameters(),
         'lr': cfg_optim['classifier']['lr'],
         'weight_decay': cfg_optim['classifier']['weight_decay']},
        {'params': model.FeatureFilter.parameters(),
         'lr': cfg_optim['feature_filter']['lr'],
         'weight_decay': cfg_optim['feature_filter']['weight_decay']},
        {'params': model.layer_fusion.parameters(),
         'lr': cfg_optim['layer_fusion']['lr'],
         'weight_decay': cfg_optim['layer_fusion']['weight_decay']},
        {'params': model.fusion_weights.parameters(),
         'lr': cfg_optim['fusion_weights']['lr'],
         'weight_decay': cfg_optim['fusion_weights']['weight_decay']},
        {'params': model.feature_fusion_projection.parameters(),
         'lr': cfg_optim['feature_fusion_projection']['lr'],
         'weight_decay': cfg_optim['feature_fusion_projection']['weight_decay']},
        {'params': model.crop_feature_extractor.parameters(),
         'lr': cfg_optim['crop_feature_extractor']['lr'],
         'weight_decay': cfg_optim['crop_feature_extractor']['weight_decay']},
    ]
    return AdamW(param_groups, weight_decay=cfg_optim.get('global_weight_decay', 3e-3))

def main():
    args = parse_args()
    cfg = load_config(args.config)

    torch.backends.cudnn.benchmark = True
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    extra_data_root = args.extra_data or cfg['paths'].get('extra_data_root', None)
    if extra_data_root:
        get_extra_data_path(extra_data_root)

    aug = cfg.get('augment', {})
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=aug.get('hflip_p', 0.5)),
        transforms.ColorJitter(
            aug.get('brightness', 0.1),
            aug.get('contrast',   0.1),
            aug.get('saturation', 0.1)
        ),
        transforms.Resize(tuple(cfg.get('image_size', [224, 224]))),
        transforms.ToTensor(),
    ])

    train_loader, val_loader, test_loader = build_loaders(
        args.trainset, args.valset, args.testset,
        paths=cfg['paths'],
        batch_size=cfg.get('batch_size', 16),
        num_workers=cfg.get('num_workers', 8),
    )

    dim_in = cfg['model'].get('dim_in', 768)
    dim_hidden = cfg['model'].get('dim_hidden', 768)
    net = RECC(dim_in, dim_hidden).to(args.device)

    optimizer = build_optimizer(net, cfg['optimizer'])

    if not args.test_only:
        epochs = args.epochs if args.epochs is not None else cfg.get('epochs', 30)
        train(net, train_loader, val_loader, optimizer, epochs, transform=transform, save_path=args.save_path)

    if test_loader is not None:
        test(net, test_loader, args.save_path)

if __name__ == "__main__":
    main()
