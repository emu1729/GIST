import json
import pickle
import pandas as pd
import numpy as np
import torch
import clip
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from networks import one_layer_net, ConcatenatedNetwork
import copy
from torchvision.transforms import Compose, Resize, CenterCrop, RandomHorizontalFlip, ToTensor, RandomResizedCrop
from PIL import Image
import argparse


parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--metadata', type=str, default=None, metavar='N',
                    help='path to metadata file')
parser.add_argument('--captions_file', type=str, default=None, metavar='N',
                    help='path to captions file')
parser.add_argument('--num_captions', type=int, default=1, metavar='N',
                    help='number of captions to match')
parser.add_argument('--clip_model', type=str, default='ViT-L/14@336px', metavar='N',
                    help='clip model')
parser.add_argument('--image_folder', type=str, default=None, metavar='N',
                    help='image folder')
parser.add_argument('--output_file', type=str, default=None, metavar='N',
                    help='where to save matched output')


class image_text_dataset(Dataset):
    def __init__(self, list_image_path, list_txt):
        self.image_path = list_image_path
        self.title = clip.tokenize(list_txt)

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        image = train_transforms(Image.open(self.image_path[idx]))  # Image from PIL module
        title = self.title[idx]
        return image, title


def run_fine_tuning(captions_file, metadata, image_folder, num_captions, clip_model, output_file):

    model, preprocess = clip.load(clip_model, device='cuda', jit=False)  # Must set jit=False for training
    clip.model.convert_weights(model)
    EPOCH = 200
    BATCH_SIZE = 20

    train_transforms = Compose([
        RandomResizedCrop(336),
        RandomHorizontalFlip(),
        preprocess,  # apply the CLIP preprocessing to the images
    ])

    captions_dict = json.load(open(captions_file, 'rb'))
    df = pd.read_csv(metadata)
    df = df[df['split'] == 'train']

    list_image_path = []
    list_txt = []
    for i, row in df.iterrows():
        for j in range(num_captions):
            caption = row['species'] + ': ' + captions_dict[row['filename']][j]
            list_txt.append(caption)
            list_image_path.append(image_folder + row['filename'])

    print(len(list_txt))
    dataset = image_text_dataset(list_image_path, list_txt)
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)  # Define your own dataloader

    # https://github.com/openai/CLIP/issues/57
    def convert_models_to_fp32(model):
        for p in model.parameters():
            p.data = p.data.float()
            p.grad.data = p.grad.data.float()

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, betas=(0.9, 0.98), eps=1e-6,
                           weight_decay=0.001)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

    # add your own code to track the training progress.
    for epoch in range(EPOCH):
        print("Training Epoch: ", epoch)
        for batch in train_dataloader:
            optimizer.zero_grad()

            images, texts = batch

            images = images.to('cuda')
            texts = texts.to('cuda')

            logits_per_image, logits_per_text = model(images, texts)

            ground_truth = torch.arange(len(images), dtype=torch.long, device='cuda')

            total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
            total_loss.backward()

            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)

        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), output_file)


if __name__ == '__main__':
    args = parser.parse_args()
    run_fine_tuning(args.captions_file, args.metadata, args.image_folder, args.num_captions, args.clip_model, args.output_file)
