from clip_featurizer import ClipFeaturizer
import pandas as pd
from PIL import Image
from tqdm import tqdm
import pickle
import torch
import argparse

parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--metadata', type=str, default=None, metavar='N',
                    help='path to metadata file')
parser.add_argument('--clip_weights', type=str, default=None, metavar='N',
                    help='path to clip weights if they exist')
parser.add_argument('--image_folder', type=str, default=None, metavar='N',
                    help='image folder')
parser.add_argument('--output_file', type=str, default=None, metavar='N',
                    help='where to save matched output')


def run_clip_embeddings(metadata, clip_weights, image_folder, output_file):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if clip_weights:
        clip_feat = ClipFeaturizer('ViT-L/14@336px', device, use_logit_scale=False, weights_path=clip_weights)
    else:
        clip_feat = ClipFeaturizer('ViT-L/14@336px', device, use_logit_scale=False)
        
    df = pd.read_csv(metadata)

    embeddings = {}

    with tqdm(total=df.shape[0]) as pbar:
        for index, row in tqdm(df.iterrows()):
            filename = image_folder + row['filename']
            im = Image.open(filename)
            embeddings[row['filename']] = clip_feat.compute_image_feature(im).detach().cpu().numpy()
            pbar.update(1)

    pickle.dump(embeddings, open(output_file, 'wb'))


if __name__ == "__main__":
    args = parser.parse_args()
    run_clip_embeddings(args.metadata, args.clip_weights, args.image_folder, args.output_file)
    


