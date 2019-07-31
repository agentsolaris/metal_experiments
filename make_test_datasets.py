import os
import pandas as pd
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--orig_data_dir', default="./data/", help='The file path of the BERT ready original dataset')
parser.add_argument('--bin_data_dir', default="./bin_data/", help='The target file path of the new binaried dataset')
parser.add_argument('--trunc_data_dir', default="./trunc_data/", help='The target file path of the new truncated 5% dataset')
parser.add_argument('--bin_emotions', default="joy,anger", help='The emotions to be binaried (comma separated)')

args = parser.parse_args()
original_data_dir = args.orig_data_dir
binaried_data_dir = args.bin_data_dir
truncated_data_dir = args.trunc_data_dir

binary_emotions = args.bin_emotions.split(",")

assert os.path.exists(original_data_dir)

if not os.path.exists(binaried_data_dir):
    os.makedirs(binaried_data_dir)

if not os.path.exists(truncated_data_dir):
    os.makedirs(truncated_data_dir)

for data_type in ["train", "dev", "test"]:
    original_data = pd.read_csv(os.path.join(original_data_dir, data_type+".tsv"), sep="\t", error_bad_lines=False)
    
    #binaried_df = original_data.loc[original_data.emotion.isin(binary_emotions),]
    #binaried_df.to_csv(os.path.join(binaried_data_dir, data_type+".tsv"), sep='\t', encoding="utf-8")
        
    truncated_df = original_data.sample(frac=0.05)
    truncated_df.to_csv(os.path.join(truncated_data_dir, data_type+".tsv"), sep='\t', encoding="utf-8")
