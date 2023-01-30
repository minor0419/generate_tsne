import matplotlib.pyplot as plt
import cv2
import time
import os
from models.i3d.extract_i3d import ExtractI3D
from utils.utils import build_cfg_path
from omegaconf import OmegaConf
import torch
import pandas as pd
import json
import glob
import numpy as np
import tsnecuda
import argparse


def generate_tsne(args_main, name):
    label_file = open('./label_define.json')
    label_dict = json.load(label_file)
    index_dict_value = list(label_dict)
    label_file.close()

    # name = '動画1'
    pkl_path = './result_video_features/' + name + '/*.pkl'
    pkl_path_list = glob.glob(pkl_path)
    pkl_path_list.sort()

    data_frame = pd.DataFrame()

    i = 0
    for data_path in pkl_path_list:
        df = pd.read_pickle(data_path, compression="tar")
        data_frame = pd.concat([data_frame, df])

    i3d_series = data_frame['i3d']['rgb']['rgb']
    label_series = data_frame['label']['rgb']['rgb']
    l_dict = label_dict['label']
    i = 0
    y = []
    for l in label_series:
        keys = [k for k, v in l_dict.items() if v == l]
        key_int = int(keys[0])
        y.append(key_int)
        i += 1
    X = []
    Y = []
    i = 0
    for x in i3d_series.values:
        a, b = x.shape
        j = 0
        for j in range(a):
            X.append(x[j])
            Y.append(y[i])
            j += 1
        i += 1
    X = np.array(X)

    # t-SNEで次元削減 15<=perplexity<=50(van der MaatenとHinton)
    # 動画5--iter 300 --neighbors 300 --perplexity 50
    tsne = tsnecuda.TSNE(n_components=2, n_iter=args.iter, verbose=1, perplexity=args.perplexity,
                         num_neighbors=args.neighbors)
    tsne_results = tsne.fit_transform(X)
    print('Success!!')

    plt.figure(figsize=(13, 7))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1],
                c=Y, cmap='jet',
                s=15, alpha=0.5)
    plt.axis('off')
    plt.colorbar()
    plt.subplot().set_title(name, fontsize=30)
    plt.savefig(name + '.png')
    # plt.show()


def main(args_main):
    print(torch.__version__)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.cuda.get_device_name(0)

    #torch.cuda.set_per_process_memory_fraction(0.1, 0)
    torch.cuda.empty_cache()
    total_memory = torch.cuda.get_device_properties(0).total_memory
    # less than 0.5 will be ok:
    tmp_tensor = torch.empty(int(total_memory * 0.099), dtype=torch.int8, device='cuda')
    del tmp_tensor
    torch.cuda.empty_cache()
    start_position = int(args_main.start) - 1
    end_position = int(args_main.end)

    # Select the feature type
    feature_type = 'i3d'

    # Load and patch the config
    args = OmegaConf.load(build_cfg_path(feature_type))
    args.stack_size = 12
    args.step_size = 1
    args.extraction_fps = 30
    args.streams = 'rgb'

    # Load the model
    extractor = ExtractI3D(args)

    label_file = open('./label_define.json')
    label_dict = json.load(label_file)
    label_file.close()

    k = 0
    movies = glob.glob('*.mp4')
    for movie in movies:
        name = os.path.basename(movie).replace('.mp4', '')
        print(name)
        json_file = open('./' + name + '.json', 'r')
        json_dict = json.load(json_file)
        json_dict_value = list(json_dict)
        json_file.close()

        input_path = './' + name + '.mp4'
        cap = cv2.VideoCapture(input_path)
        frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        data_frame = pd.DataFrame()
        df_i3d = pd.DataFrame()

        for area in json_dict_value:
            area_data = json_dict[area]
            area_annotation = area_data['annotations']
            area_label = area_annotation['label']
            area_segments = area_annotation['segments']
            area_segments_start = area_segments[0]['time']
            area_segments_end = area_segments[1]['time']
            x1 = int(area_segments[0]['rect'][0])
            y1 = int(area_segments[0]['rect'][1])
            x2 = int(area_segments[0]['rect'][2])
            y2 = int(area_segments[0]['rect'][3])
            new_width = int(x2 - x1)
            new_height = int(y2 - y1)
            output_file = './result_video/' + name + '/' + area + '_' + area_label + '.mp4'
            output_pkl = './result_video_features/' + name + '/' + area + '_' + area_label + '.pkl'
            output_npy = './result_video_features/' + name + '/' + area + '_' + area_label + '.npy'
            os.makedirs('./result_video/' + name, exist_ok=True)
            os.makedirs('./result_video_features/' + name, exist_ok=True)

            new_width2 = int(new_width / 1)
            new_height2 = int(new_height / 1)

            print(output_file)
            fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            writer = cv2.VideoWriter(output_file, fmt, fps, (new_width2, new_height2))
            cap = cv2.VideoCapture(input_path)
            i = 0
            j = 0
            while True:
                ret, frame = cap.read()
                current_time = i / fps
                if not ret:
                    break
                if area_segments_start < current_time < area_segments_end:
                    cliped_frame = frame[x1: x2, y1: y2]
                    cliped_frame = cv2.resize(cliped_frame, (new_width2, new_height2))

                    writer.write(cliped_frame)

                    j += 1
                else:
                    if current_time > area_segments_end:
                        break
                i += 1

            cap.release()
            writer.release()

            try:
                time_sta = time.time()
                feature_dict = extractor.extract(output_file)
                feature_dict_rgb = feature_dict['rgb']
                x, y = feature_dict_rgb.shape
                if y != 1024:
                    print(f'********** {output_npy:s} is ignoed. **********')
                    raise ValueError("error!")
                else:
                    time_end = time.time()
                    tim = time_end - time_sta
                    print(tim)

                    labels = label_dict['label']
                    for i in range(len(labels)):
                        label = labels[str(i)]
                        if label == area_label:
                            list_search = label

                    i3d_series = pd.Series(feature_dict)
                    df_i3d['i3d'] = i3d_series
                    df_i3d['label'] = list_search
                    df_i3d.to_pickle(output_pkl, "tar")

                    print(f'Name {output_file:s} is Finished!!!')

            except:
                print(f'********** {output_npy:s} is ignoed. **********')
        k += 1
        generate_tsne(args_main, name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--start', help="start position", default=1)
    parser.add_argument('--end', help="end position", default=4000)
    parser.add_argument('--iter', help="start position", default=300)
    parser.add_argument('--perplexity', help="end position", default=50)
    parser.add_argument('--neighbors', help="end position", default=300)
    args = parser.parse_args()

    os.makedirs('./result_video/', exist_ok=True)
    os.makedirs('./result_video_features/', exist_ok=True)

    main(args)
