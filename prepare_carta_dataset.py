import util.file_handler
import cv2
import os
from tqdm import tqdm
import numpy as np
import asyncio
import shutil

from util.video_handler import VideoHandler


async def write(frame, path):
    cv2.imwrite(frame, path)


def load_video(path):
    video_in = cv2.VideoCapture(path)
    success = True
    while success:
        success, frame = video_in.read()
        if success:
            yield frame
    video_in.release()


async def main():

    gt_base_path = 'dataset/gt/'

    ground_truth = [{} for _ in range(26)]

    print('Loading and indexing ground truth')
    for i in range(1, 25):
        ground_truth[i] = util.file_handler.load_json(gt_base_path + f'video_{i}_gt.json')
        for image in ground_truth[i]['images']:
            image['annotations'] = []

        for annotation in ground_truth[i]['annotations']:
            ground_truth[i]['images'][annotation['image_id'] - 1]['annotations'].append(annotation)

    dataset = {
        'info': {},
        'licenses': [],
        'images': [],
        'annotations': [],
        'categories': [
            {
                'id': 1,
                'supercategory': 'none',
                'name': 'head'
            }
        ]
    }

    base_path = 'dataset/carta/'

    if not os.path.isdir(base_path):
        os.makedirs(base_path, exist_ok=True)

    annotation_index = 1
    image_index = 0

    bgr = np.zeros(3)

    for i in range(1, 25):
        for image in tqdm(ground_truth[i]['images']):
            frame_number = int(image['file_name'].split('_')[1].split('.')[0])

            image_spec = {
                'id': image_index,
                'file_name': f'{image_index}.jpg',
                'width': 1920,
                'height': 1080
            }
            dataset['images'].append(image_spec)
            shutil.copyfile(f'dataset/split_pane/video_{i}/pane_3/frame_{frame_number}.jpg',
                            f'{base_path}{image_index}.jpg')

            for annotation in image['annotations']:
                bbox = annotation['bbox']
                bbox[0] -= 1920
                bbox[1] -= 1080
                new = {
                    'id': annotation_index,
                    'image_id': image_index,
                    'area': bbox[2] * bbox[3],
                    'bbox': bbox,
                    'iscrowd': 0,
                    'category_id': 1
                }
                dataset['annotations'].append(new)

                annotation_index += 1

            image_index += 1

    util.file_handler.write_json(base_path + 'annotations.json', dataset)

    print(f'Standard: {bgr/image_index}')
    print(f'Total frames: {image_index}')
    print(f'Total annotations: {annotation_index}')


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())

