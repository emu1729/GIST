# GIST: Generating Image Specific Text
Code for ArXiv paper ["Generating Image Specific Text For Fine-Grained Classification"](https://arxiv.org/abs/2307.11315)

## Setting up Environment
You can set up a conda environment with required packages for GIST as follows:
```bash
conda create -n gist python=3.9.16
conda activate gist
conda install --file requirements.txt
```

## Setting up Datasets
All datasets are located in the `datasets/` directory. Please download images for each dataset into an `images/` directory in each dataset folder. The download links are below:

* **aircraft**: [https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz)
* **CUB**: [https://data.caltech.edu/records/20098](https://data.caltech.edu/records/20098)
* **flower**: [https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz)
* **fitzpatrick**: [https://github.com/mattgroh/fitzpatrick17k](https://github.com/mattgroh/fitzpatrick17k)

We provide example 5, 3, and 1 shot metadata files for each dataset as well. The Fitzpatrick40 metadata file (`datasets/fitzpatrick40/metadata.csv`) can be used to create our cleaned Fitzpatrick40 dataset from the original Fitzpatrick17k dataset.

## Caption Matching
We provide json files of our top-5 matched captions for training images of the aircraft, CUB, flower, and fitzpatrick datasets. We additionally provide an example script for how to match captions `run_caption_matching_example.sh` in case you want to try matching captions for a novel dataset.

## Fine-Grained Classification
We provide scripts to run fine-grained all, 5, 3, and 1 shot classification on each of the aircraft, CUB, flower, and fitzpatrick datasets. As an example, if a user wants to run 5 shot fine-grained classification on the aircraft dataset, they would first run contrastive fine-tuning

```bash
CUDA_VISIBLE_DEVICES=0 python contrastive_training_clip.py --metadata datasets/aircraft/metadata_5shot.csv --captions_file datasets/aircraft/captions_top5.json --num_captions 4 --image_folder datasets/aircraft/images/ --output_file aircraft_5shot
```

Once the contrastive fine-tuning is run, you can create clip embeddings for images using a fine-tuned network as follows

```bash
CUDA_VISIBLE_DEVICES=0 python create_clip_embeddings.py --metadata datasets/aircraft/metadata_5shot.csv --clip_weights aircraft_5shot_epoch_39.pt --image_folder datasets/aircraft/images/ --output_file datasets/aircraft/aircraft_5shot_embeddings.pkl
```

Finally, we can learn a linear image probe for the fine-tuned CLIP network as follows

```bash
CUDA_VISIBLE_DEVICES=0 python linear_probe.py --image_embedding_file datasets/aircraft/aircraft_5shot_embeddings.pkl --metadata datasets/aircraft/metadata_5shot.csv
```



