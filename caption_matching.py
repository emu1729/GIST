import torch
from PIL import Image
import open_clip
import json
import pandas as pd
import numpy as np
import tqdm
import argparse

parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--model_name', type=str, default='ViT-B-32', metavar='N',
                    help='model name')
parser.add_argument('--long_captions', type=str, default=None, metavar='N',
                    help='path to long class captions')
parser.add_argument('--metadata', type=str, default=None, metavar='N',
                    help='metadata file')
parser.add_argument('--image_folder', type=str, default=None, metavar='N',
                    help='image folder')
parser.add_argument('--class_file', type=str, default=None, metavar='N',
                    help='includes all classes')
parser.add_argument('--output_file', type=str, default=None, metavar='N',
                    help='where to save matched output')


def run_matching(model_name, long_captions, metadata, image_folder, class_file, output_file):
    # Load Model
    print("Loading Model ... ")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer(model_name)

    # Load Captions
    print("Loading Captions ... ")
    captions_dict = json.load(open(long_captions, 'rb'))
    all_captions = []
    reverse_captions_dict = {}
    for key in captions_dict.keys():
        all_captions.extend(captions_dict[key])
        captions = captions_dict[key]
        for caption in captions:
            reverse_captions_dict[caption] = key

    # Load Images
    print("Loading Images ... ")
    df = pd.read_csv(metadata)
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

    print("Compute Labels...")
    with open(class_file, "r") as file:
        lines = file.readlines()

    lines = [line.split('.')[1].lower().replace("_", "-") for line in lines]

    print("Matching Images and Text...")
    filename_caption_dict = {}
    for index, row in tqdm.tqdm(df_train.iterrows()):
        filename = image_folder + row['filename']
        label = row['label']
        if row['filename'] not in filename_caption_dict:

            image = preprocess(Image.open(filename)).unsqueeze(0)

            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                specific_text_features = specific_text_features_dict[lines[label]]

                specific_text_probs = (100.0 * image_features @ specific_text_features.T).softmax(dim=-1)
                specific_text_probs = torch.flatten(specific_text_probs).cpu().detach().numpy()

            specific_sorted_indices = np.argsort(-specific_text_probs)

            top_specific_indices = specific_sorted_indices[:5]
            print("Label: ", row['label'])

            print([captions_dict[lines[label]][ind] for ind in top_specific_indices])
            filename_caption_dict[row['filename']] = [captions_dict[lines[label]][ind] for ind in top_specific_indices]
            json.dump(filename_caption_dict, open(output_file, 'w'))

    json.dump(filename_caption_dict, open(output_file, 'w'))


if __name__ == '__main__':
    args = parser.parse_args()
    run_matching(args.model_name, args.long_captions, args.metadata, args.image_folder, args.class_file, args.output_file)