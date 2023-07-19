# Make sure to download CUB images into an /images directory in datasets/CUB

python caption_matching.py --long_captions datasets/CUB/long_captions.json --metadata datasets/CUB/metadata.csv --image_folder datasets/CUB/images/ --class_file datasets/CUB/classes.txt --output_file datasets/CUB/matched_long.json

