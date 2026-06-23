import random

import numpy as np
from typing_extensions import final
from .Unet import Sequel_U_net as U_Net
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from modules.Classifier import DeepMLPClassifier as ConvClassifier
import copy

class CropFeatureExtractor(nn.Module):
    def __init__(self, output_dim=768, seq_len=50):
        super(CropFeatureExtractor, self).__init__()
        self.output_dim = output_dim
        self.seq_len = seq_len

        self.cnn_features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.feature_projection = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, output_dim),
            nn.LayerNorm(output_dim)
        )
        self.seq_adapter = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim)
        )

    def forward(self, crop_img_list):
        device = next(self.parameters()).device
        crop_tensors = []
        for img in crop_img_list:
            if isinstance(img, Image.Image):
                tensor = torch.tensor(np.array(img)).float() / 255.0
                if tensor.dim() == 2:
                    tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
                elif tensor.dim() == 3:
                    tensor = tensor.permute(2, 0, 1)
            else:
                tensor = img
            crop_tensors.append(tensor)

        batch_crops = torch.stack(crop_tensors).to(device)
		
        cnn_features = self.cnn_features(batch_crops)
        cnn_features = cnn_features.squeeze(-1).squeeze(-1)
        projected_features = self.feature_projection(cnn_features)
        if projected_features.shape[0] > self.seq_len:
            num_groups = projected_features.shape[0] // self.seq_len
            remainder = projected_features.shape[0] % self.seq_len
            grouped_features = projected_features[:num_groups * self.seq_len].view(
                num_groups, self.seq_len, self.output_dim
            )
            pooled_features = torch.mean(grouped_features, dim=0)
            if remainder > 0:
                remaining_features = projected_features[num_groups * self.seq_len:]
                padding = torch.zeros(self.seq_len - remainder, self.output_dim, device=device)
                remaining_padded = torch.cat([remaining_features, padding], dim=0)
                pooled_features = (pooled_features + remaining_padded) / 2
            final_features = pooled_features.unsqueeze(0)
        elif projected_features.shape[0] < self.seq_len:
            padding_size = self.seq_len - projected_features.shape[0]
            last_feature = projected_features[-1:].repeat(padding_size, 1)
            padded_features = torch.cat([projected_features, last_feature], dim=0)
            final_features = padded_features.unsqueeze(0)
        else:
            final_features = projected_features.unsqueeze(0)
        final_features = self.seq_adapter(final_features)
        return final_features

class RECC(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RECC, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.Unet1 = U_Net(self.in_channel, self.out_channel)
        self.Classifier = ConvClassifier()
        self.clip_model = CLIPModel.from_pretrained("modules/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("modules/clip-vit-base-patch32")
        self.ChannelAttention = ChannelAttention(channels=in_channel)
        self.SpatialAttention = SpatialAttention()
        self.FeatureFilter = SequenceFeatureFilter(in_channel, in_channel)

        self.crop_feature_extractor = CropFeatureExtractor(
            output_dim=in_channel,
            seq_len=50
        )

        self.layer_fusion = LayerWeightedFusion(
            num_layers=13,
            hidden_size=768,
            fusion_method='weighted_sum'
        )
        self.fusion_weights = nn.Sequential(
            nn.Linear(in_channel * 2, in_channel),
            nn.ReLU(),
            nn.Linear(in_channel, 2),
            nn.Softmax(dim=-1)
        )
        self.feature_fusion_projection = nn.Sequential(
            nn.Linear(in_channel * 2, in_channel * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_channel * 2, in_channel),
            nn.LayerNorm(in_channel)
        )

    def backbone(self, img):
        inputs = self.clip_processor(images=img, return_tensors="pt", do_rescale=False).to(img.device)
        img_outputs = self.clip_model.vision_model(**inputs, output_hidden_states=True)
        fused_features = self.layer_fusion(img_outputs.hidden_states)
        return fused_features

    def crop(self, img):
        crop_size = [2, 4, 8, 16, 32, 64, 128]
        crop_num = [120, 90, 50, 40, 20, 10, 2]
        crop_img = []

        for i in range(len(crop_size)):
            size = crop_size[i]
            num = crop_num[i]

            if size == 2:
                step = max(2, (224 - size) // int(np.sqrt(num * 0.8)))
                coords = []
                for x in range(0, 224 - size, step):
                    for y in range(0, 224 - size, step):
                        coords.append((x, y))
                        if len(coords) >= num * 0.7:
                            break
                    if len(coords) >= num * 0.7:
                        break
                while len(coords) < num:
                    x = random.randint(0, 224 - size)
                    y = random.randint(0, 224 - size)
                    coords.append((x, y))

            elif size == 4:
                step = max(3, (224 - size) // int(np.sqrt(num * 0.6)))
                coords = []
                for x in range(0, 224 - size, step):
                    for y in range(0, 224 - size, step):
                        coords.append((x, y))
                        if len(coords) >= num * 0.6:
                            break
                    if len(coords) >= num * 0.6:
                        break
                while len(coords) < num:
                    x = random.randint(0, 224 - size)
                    y = random.randint(0, 224 - size)
                    coords.append((x, y))

            elif size in [8, 16]:
                step = max(4, (224 - size) // int(np.sqrt(num * 0.5)))
                coords = []
                for x in range(0, 224 - size, step):
                    for y in range(0, 224 - size, step):
                        coords.append((x, y))
                        if len(coords) >= num * 0.5:
                            break
                    if len(coords) >= num * 0.5:
                        break
                while len(coords) < num:
                    x = random.randint(0, 224 - size)
                    y = random.randint(0, 224 - size)
                    coords.append((x, y))
            else:
                coords = [(random.randint(0, 224 - size), random.randint(0, 224 - size))
                          for _ in range(num)]
            for x, y in coords:
                if isinstance(img, torch.Tensor):
                    cropped = img[:, y:y + size, x:x + size]
                    if size != 32:
                        cropped = F.interpolate(cropped.unsqueeze(0), size=(32, 32),
                                                mode='bilinear', align_corners=False).squeeze(0)
                else:
                    cropped = img.crop((x, y, x + size, y + size))
                    if size != 64:
                        cropped = cropped.resize((64, 64), Image.LANCZOS)

                crop_img.append(cropped)

        return crop_img

    def forward(self, img, e_img, label):
        batch_size = img.shape[0]
        all_crop_features = []
        for i in range(batch_size):
            single_img = img[i]
            crop_img = self.crop(single_img)
            crop_features = self.crop_feature_extractor(crop_img)
            all_crop_features.append(crop_features)
        crop_features = torch.cat(all_crop_features, dim=0)

        feature = self.backbone(img)
        e_feature = None
        if e_img is not None:
            e_feature = self.backbone(e_img)

        rcon_feature, recon_crop_feature  = self.Unet1(feature, crop_features)
        mask = (label == 0) if label is not None else torch.ones(rcon_feature.shape[0], dtype=torch.bool,
                                                                 device=rcon_feature.device)

        filtered_feature = torch.zeros_like(rcon_feature)
        filtered_feature = self.FeatureFilter(rcon_feature)
        recon_crop_feature = self.FeatureFilter(recon_crop_feature)
        concat_features = torch.cat([filtered_feature, recon_crop_feature], dim=-1)
        fused_features = self.feature_fusion_projection(concat_features)
        predict = self.Classifier(fused_features)
        return {
            "feature": feature,
            "e_feature": e_feature,
            "rcon_feature": rcon_feature,
            "predict": predict,
            "filtered_feature": filtered_feature,
            "crop_features": crop_features,
            "recon_crop_feature": recon_crop_feature,
            "fused_features": fused_features
        }


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_permuted = x.permute(0, 2, 1)
        avg_out = self.fc(self.avg_pool(x_permuted).squeeze(-1))
        max_out = self.fc(self.max_pool(x_permuted).squeeze(-1))
        attention = self.sigmoid(avg_out + max_out).unsqueeze(-1)
        out = x_permuted * attention
        return out.permute(0, 2, 1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_permuted = x.permute(0, 2, 1)
        avg_out = torch.mean(x_permuted, dim=1, keepdim=True)
        max_out, _ = torch.max(x_permuted, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(concat))
        out = x_permuted * attention
        return out.permute(0, 2, 1)


class LayerWeightedFusion(nn.Module):
    def __init__(self, num_layers, hidden_size, fusion_method='attention'):
        super(LayerWeightedFusion, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.fusion_method = fusion_method

        if fusion_method == 'weighted_sum':
            self.layer_weights = nn.Parameter(torch.ones(num_layers))

        elif fusion_method == 'attention':
            self.attention_weights = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, 1)
            )

        elif fusion_method == 'mlp_fusion':
            self.fusion_mlp = nn.Sequential(
                nn.Linear(num_layers * hidden_size, hidden_size * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size * 2, hidden_size)
            )

    def forward(self, hidden_states_tuple):
        hidden_states = torch.stack(hidden_states_tuple, dim=0)

        if self.fusion_method == 'weighted_sum':
            return self._weighted_sum_fusion(hidden_states)
        elif self.fusion_method == 'attention':
            return self._attention_fusion(hidden_states)
        elif self.fusion_method == 'mlp_fusion':
            return self._mlp_fusion(hidden_states)
        elif self.fusion_method == 'adaptive_fusion':
            return self.adaptive_fusion(hidden_states)

    def _weighted_sum_fusion(self, hidden_states):
        weights = F.softmax(self.layer_weights, dim=0)
        weighted_states = weights.view(-1, 1, 1, 1) * hidden_states
        fused = torch.sum(weighted_states, dim=0)
        return fused

    def _attention_fusion(self, hidden_states):
        num_layers, batch_size, seq_len, hidden_size = hidden_states.shape
        attention_scores = []
        for i in range(num_layers):
            layer_features = hidden_states[i]
            global_features = torch.mean(layer_features, dim=1)
            score = self.attention_weights(global_features)
            attention_scores.append(score)
        attention_scores = torch.stack(attention_scores, dim=1)
        attention_weights = F.softmax(attention_scores, dim=1)
        attention_weights = attention_weights.unsqueeze(-1)
        hidden_states_permuted = hidden_states.permute(1, 0, 2, 3)
        fused = torch.sum(hidden_states_permuted * attention_weights, dim=1)

        return fused

    def _mlp_fusion(self, hidden_states):
        num_layers, batch_size, seq_len, hidden_size = hidden_states.shape
        flattened = hidden_states.permute(1, 2, 0, 3).contiguous()
        flattened = flattened.view(batch_size, seq_len, -1)
        fused = self.fusion_mlp(flattened)
        return fused


class SequenceFeatureFilter(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, in_channels),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(in_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.self_attention = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.feed_forward = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(in_channels * 4, in_channels)
        )

    def forward(self, x):
        batch_size, seq_len, in_channels = x.shape
        x_conv = x.permute(0, 2, 1)
        feat = self.conv1(x_conv)
        feat = F.silu(feat)
        channel_pool = F.adaptive_avg_pool1d(feat, 1).squeeze(-1)
        channel_weights = self.channel_attention[1:](channel_pool).unsqueeze(-1)
        feat = feat * channel_weights
        spatial_weights = self.spatial_attention(feat)
        feat = feat * spatial_weights
        feat_seq = feat.permute(0, 2, 1)
        x_seq = x_conv.permute(0, 2, 1)
        attn_out, _ = self.self_attention(feat_seq, feat_seq, feat_seq)
        feat_seq = self.norm1(feat_seq + attn_out)
        ff_out = self.feed_forward(feat_seq)
        feat_seq = self.norm2(feat_seq + ff_out)
        feat_final = feat_seq.permute(0, 2, 1)
        output = self.conv2(feat_final)
        output = output.permute(0, 2, 1)

        return output
