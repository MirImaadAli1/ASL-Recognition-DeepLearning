import os
import cv2
import glob
import pandas
import numpy as np
from tqdm import tqdm
import argparse
import re
from functools import partial
from multiprocessing import Pool


def csv2dict(anno_path, dataset_type):
    info_dict = {}
    dataset_prefix = anno_path  # This should be a CSV file, not a directory

    if not os.path.isfile(dataset_prefix):  
        raise FileNotFoundError(f"Annotation file not found: {dataset_prefix}")

    df = pandas.read_csv(dataset_prefix)  # Load CSV

    for idx, row in df.iterrows():
        fileid = row['sentence_name']  # The file ID should match the video ID
        label = row['label']
        
        # Construct the correct path to the frames, assuming structure is ./videos/{sentence_meaning}/{fileid}/frames/
        frames_path = f"../videos/{fileid}/"
        # print(f"Looking for frames in: {frames_path}")
        
        
        # Check if the path exists and count the .jpg files inside it
        num_frames = len(glob.glob(os.path.join(frames_path, "*.jpg")))
        # print(f"Frames for {fileid}: {num_frames}")

        info_dict[idx] = {
            'prefix': '../videos/',
            'fileid': fileid,
            'folder': fileid,
            'signer': 'Unknown',
            'label': label,
            'num_frames': num_frames,
            'original_info': f"{fileid}|{label}",
        }
    
    return info_dict




def generate_gt_stm(info, save_path):
    with open(save_path, "w") as f:
        for k, v in info.items():
            # Check if required keys exist
            if 'fileid' not in v or 'label' not in v:
                print(f"Skipping {k}: Missing fileid or label")
                continue
            
            # Write each entry
            f.writelines(f"{v['fileid']} 1 {v['signer']} 0.0 1.79769e+308 {v['label']}\n")


def sign_dict_update(total_dict, info):
    # Add the blank token explicitly
    # total_dict[' '] = 0  # or 1 depending on your use case
    
    for k, v in info.items():
        if not isinstance(k, int):  # Ensure keys are integers
            continue
        
        split_label = v['label'].split()  # Split label into individual glosses
        for gloss in split_label:
            if gloss not in total_dict:
                total_dict[gloss] = 1
            else:
                total_dict[gloss] += 1
    return total_dict


def resize_img(img_path, dsize='210x260px'):
    dsize = tuple(int(res) for res in re.findall("\d+", dsize))
    img = cv2.imread(img_path)
    img = cv2.resize(img, dsize, interpolation=cv2.INTER_LANCZOS4)
    return img


def resize_dataset(video_idx, dsize, info_dict):
    # for key, value in info_dict.items():
    #     try:
    #         print(f"Prefix for {key}: {value['prefix']}")
    #     except KeyError as e:
    #         print(f"KeyError: {e} for {key}")

    info = info_dict[video_idx]  # video_idx should now be a string (video name)
    if 'prefix' not in info_dict:
        info_dict['prefix'] = '../videos/'  # Assign default prefix
    if 'prefix' in info_dict:
        img_list = glob.glob(f"{info_dict['prefix']}/{info['folder']}/{info['fileid']}/*.jpg")
    else:
        print(f"Missing 'prefix' for file: {info['fileid']}")
        # Handle the case where 'prefix' is missing (maybe skip or set a default value)
    for img_path in img_list:
        rs_img = resize_img(img_path, dsize=dsize)
        rs_img_path = img_path.replace("210x260px", dsize)
        rs_img_dir = os.path.dirname(rs_img_path)
        if not os.path.exists(rs_img_dir):
            os.makedirs(rs_img_dir)
            cv2.imwrite(rs_img_path, rs_img)
        else:
            cv2.imwrite(rs_img_path, rs_img)


def run_mp_cmd(processes, process_func, process_args):
    with Pool(processes) as p:
        outputs = list(tqdm(p.imap(process_func, process_args), total=len(process_args)))
    return outputs


def run_cmd(func, args):
    return func(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Data process for Visual Alignment Constraint for Continuous Sign Language Recognition.')
    parser.add_argument('--dataset', type=str, default='how2sign',
                        help='save prefix')
    parser.add_argument('--dataset-root', type=str, default='./videos',
                        help='path to the dataset')
    parser.add_argument('--annotation-prefix', type=str, default='splits/{}.csv',
                     help='annotation prefix')  
    parser.add_argument('--output-res', type=str, default='256x256px',
                        help='resize resolution for image sequence')
    parser.add_argument('--process-image', '-p', action='store_true',
                        help='resize image')
    parser.add_argument('--multiprocessing', '-m', action='store_true',
                        help='whether adopts multiprocessing to accelerate the preprocess')

    args = parser.parse_args()
    mode = ["dev", "test", "train"]
    
    sign_dict = dict()
    if not os.path.exists(f"./{args.dataset}"):
        os.makedirs(f"./{args.dataset}")
    for md in mode:
        # generate information dict
        information = csv2dict(f"{args.dataset_root}/{args.annotation_prefix.format(md)}", dataset_type=md)
        np.save(f"./{args.dataset}/{md}_info.npy", information)
        # update the total gloss dict
        sign_dict_update(sign_dict, information)
        # generate groundtruth stm for evaluation
        generate_gt_stm(information, f"./{args.dataset}/{args.dataset}-groundtruth-{md}.stm")
        # resize images
        video_index = list(information.keys())   # ✅ Get actual video names
        print(f"Resize image to {args.output_res}")
        # Print the first 10 entries of the 'information' dictionary
        print("INFO_DICT CONTENT (First 10 entries):", dict(list(information.items())[:10]))

        if args.process_image:
            if args.multiprocessing:
                run_mp_cmd(10, partial(resize_dataset, dsize=args.output_res, info_dict=information), video_index)
            else:
                for idx in tqdm(video_index):
                    run_cmd(partial(resize_dataset, dsize=args.output_res, info_dict=information), idx)
    sign_dict = sorted(sign_dict.items(), key=lambda d: d[0])
    save_dict = {}
    for idx, (key, value) in enumerate(sign_dict):
        save_dict[key] = [idx + 1, value]
    np.save(f"./{args.dataset}/gloss_dict.npy", save_dict)