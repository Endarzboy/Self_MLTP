"""
A Python code to extract and split the UCF101 videos.

"""

import os

import cv2
import argparse
import numpy as np
from tqdm import tqdm

def split(video_dir: str, split_dir: str, size=0.2, save_video_info=False):

    for folder in ['train', 'test']:
        folder_path = os.path.join(split_dir, folder)

        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
            print("Folder {} is created".format(folder_path))

    train_set = []
    test_set = []
    classes = os.listdir(video_dir)
    num_classes = len(classes)

    for class_index, classname in enumerate(classes):
        videos = os.listdir(os.path.join(video_dir, classname))

        np.random.shuffle(videos)
        split_size = int(len(videos) * float(size))

        for i in range(2):
            part = ['train', 'test'][i]
            class_dir = os.path.join(split_dir, part, classname)
            if not os.path.exists(class_dir):
                os.mkdir(class_dir)

        # Traverse each video and extract the image frame of each video
        for i in tqdm(range(len(videos)), desc='[%d/%d]%s' % (class_index + 1, num_classes, classname)):
            video_path = os.path.join(video_dir, classname, videos[i])
            video_fd = cv2.VideoCapture(video_path)

            if not video_fd.isOpened():
                print("Skipped: {}".format(video_path))
                continue

            video_type = 'test' if i <= split_size else 'train'  # split_size = 30 (0.3%)

            frame_index = 0
            success, frame = video_fd.read()

            video_name = videos[i].rsplit('.')[0]

            # Retrieve video information
            # 1: get the duration of each video
            total_frame = int(video_fd.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = video_fd.get(cv2.CAP_PROP_FPS)

            duration_in_seconds = int(total_frame / fps)

            while success:
                img_path = os.path.join(split_dir, video_type, classname, video_name)
                if not os.path.exists(img_path):
                    os.mkdir(img_path)

                frame_path = os.path.join(img_path, "%s_%d.jpg" % (video_name, frame_index))
                cv2.imwrite(frame_path, frame)
                if save_video_info:
                    info = [classname, video_name, frame_path, str(duration_in_seconds), str(int(fps)),
                            str(total_frame)]
                else:
                    info = [classname, video_name, frame_path]
                # Save the video frame information
                if video_type == 'test':
                    test_set.append(info)
                else:
                    train_set.append(info)

                frame_index += 1
                success, frame = video_fd.read()

            video_fd.release()

        # Save the training set and test set dataset to a file for easy writing to dataloader
        dataset = [train_set, test_set]
        names = ['train', 'test']

        for i in range(2):
            with open(split_dir + '/' + names[i] + '.csv', 'w') as f:
                f.write('\n'.join([','.join(line) for line in dataset[i]]))


def parse_args():
    parser = argparse.ArgumentParser(usage='python3 make_train_test.py -i path/to/UCF -o path/to/output -s 0.2')
    parser.add_argument('-i', '--video_dir', help='path to UCF dataset')
    parser.add_argument('-o', '--split_dir', help='path to output')
    parser.add_argument('-s', '--size', help='ratio of test sets')

    arg = parser.parse_args()

    return arg


if __name__ == "__main__":
    import timeit

    start = timeit.default_timer()

    args = parse_args()

    split(**vars(args))
    print("Time taken to Transform the data is: {} minutes.".format(int(timeit.default_timer() - start) / 60))

    # python3 dataset/extract_video_frames.py -i "UCF/fine_tuning_ucf101" -o "UCF/fine_tuning_ucf101_frames" -s 0.2
