# GIST: Generating Image Specific Text
Code for ArXiv paper "Generating Image Specific Text For Fine-Grained Classification"

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
* **fitzpatrick**: 

## Caption Matching
We provide json files of our top-5 matched captions for training images of the aircraft, CUB, flower, and fitzpatrick datasets. However, we additionally provide an example script for how to match captions `run_caption_matching_example.sh` in case you want to try matching captions for a novel dataset.

## Fine-Grained Classification




