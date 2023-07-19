import torch
from PIL import Image
import open_clip
import json
import pandas as pd
import numpy as np
import tqdm


# Load Model
print("Loading Model ... ")
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Load Captions
print("Loading Captions ... ")
captions_dict = json.load(open('/data/ddmg/brainGAN/captions_clip.json', 'rb'))
all_captions = []
reverse_captions_dict = {}
for key in captions_dict.keys():
    all_captions.extend(captions_dict[key])
    captions = captions_dict[key]
    for caption in captions:
        reverse_captions_dict[caption] = key

# Load Images
print("Loading Images ... ")
df = pd.read_csv('/data/ddmg/brainGAN/conditional-contrastive-networks/data/fitzpatrick_forty_label_cleaned.csv')
df = pd.read_csv('/data/ddmg/brainGAN/CUB200/cub_meta.csv')
df = pd.read_csv('/data/ddmg/xray_data/datasets/flower/splits/flowers_meta.csv')
df = pd.read_csv('/data/ddmg/xray_data/datasets/aircraft/splits/aircraft_meta.csv')
df_train = df[df['split'] == 'train']

text = tokenizer(all_captions)

specific_text = {}
for key in captions_dict.keys():
    specific_text[key] = tokenizer(captions_dict[key])

print("Computing Text Features...")
specific_text_features_dict = {}
with torch.no_grad(), torch.cuda.amp.autocast():
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    for key in captions_dict.keys():
        specific_text_features_dict[key] = model.encode_text(specific_text[key])
        specific_text_features_dict[key] /= specific_text_features_dict[key].norm(dim=-1, keepdim=True)

print("Compute Labels for CUB...")
with open('/data/ddmg/brainGAN/CUB200/classes.txt', "r") as file:
    lines = file.readlines()

lines = [line.split('.')[1].lower().replace("_", "-") for line in lines]

print("Compute Labels for Flowers...")
with open('/data/ddmg/xray_data/datasets/flower/concepts/classes.txt', "r") as file:
    lines = file.readlines()

lines = [line.lower().strip() for line in lines]

print("Compute Labels for Aircraft...")
with open('/data/ddmg/xray_data/datasets/aircraft/concepts/classes.txt', "r") as file:
    lines = file.readlines()
lines = [line.strip() for line in lines]


print("Matching Images and Text...")
#filename_caption_dict = json.load(open('/data/ddmg/brainGAN/CUB200/filename_caption.json', 'rb'))
filename_caption_dict = {}
accurate = 0
for index, row in tqdm.tqdm(df_train.iterrows()):
    #filename = '/data/ddmg/xray_data/fitzpatrick17k/' + row['filename']
    #filename = '/data/ddmg/brainGAN/CUB200/images/' + row['filename']
    #filename = '/data/ddmg/xray_data/datasets/flower/jpg/' + row['filename']
    filename = '/data/ddmg/xray_data/LaBo/datasets/aircraft/images/' + row['filename']
    label = row['label']
    if row['filename'] not in filename_caption_dict:

        image = preprocess(Image.open(filename)).unsqueeze(0)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            text_probs = torch.flatten(text_probs).cpu().detach().numpy()

            specific_text_features = specific_text_features_dict[lines[label]]

            specific_text_probs = (100.0 * image_features @ specific_text_features.T).softmax(dim=-1)
            specific_text_probs = torch.flatten(specific_text_probs).cpu().detach().numpy()

        sorted_indices = np.argsort(-text_probs)
        specific_sorted_indices = np.argsort(-specific_text_probs)

        top_indices = sorted_indices[:5]
        top_values = text_probs[top_indices]
        top_specific_indices = specific_sorted_indices[:5]
        top_specific_values = specific_text_probs[top_specific_indices]
        # print("Top Prob: ", top_values)
        # print("Top Indices: ", top_indices)
        # print("Max Caption: ", [all_captions[ind] for ind in top_indices])
        # print("Max Disease: ", [reverse_captions_dict[all_captions[ind]] for ind in top_indices])
        #
        # print("Top Specific Prob: ", top_specific_values)
        # print("Top Specific Indices: ", top_specific_indices)
        # print("Max Specific Caption: ", [captions_dict[label][ind] for ind in top_specific_indices])
        # print("Max Disease: ", [reverse_captions_dict[captions_dict[label][ind]] for ind in top_specific_indices])
        #
        print("Label: ", row['label'])
        # print("Filename", row['filename'])

        if label == top_indices[0]:
            accurate += 1
            print(accurate)

        print([captions_dict[lines[label]][ind] for ind in top_specific_indices])
        filename_caption_dict[row['filename']] = [captions_dict[lines[label]][ind] for ind in top_specific_indices]
        json.dump(filename_caption_dict, open('/data/ddmg/xray_data/datasets/aircraft/aircraft_short_captions_top5.json', 'w'))

json.dump(filename_caption_dict, open('/data/ddmg/xray_data/datasets/aircraft/aircraft_short_captions_top5.json', 'w'))
print(accurate)
print(df_train.shape)