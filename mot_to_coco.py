import cv2
import numpy as np
from tqdm import tqdm

import util.file_handler


def convert(path):

    dataset = {
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

    with open(path) as fd:
        annotation_lines = fd.readlines()

    annotation_id = 1
    rgb_sum = np.zeros(3)

    for i in range(1, int(annotation_lines[-1].split(',')[0]) + 1):
        dataset['images'].append({
            'file_name': f'{i:06d}.jpg',
            'height': 1080,
            'width': 1920,
            'id': i - 1
        })

    for line in tqdm(annotation_lines):
        frame, _id, bbx, bby, bbw, bbh, ignore, *_ = line.split(',')
        dataset['annotations'].append({
            'area': float(bbw) * float(bbh),
            'bbox': [float(bbx), float(bby), float(bbw), float(bbh)],
            'category_id': 1,
            'iscrowd': 0,
            'id': annotation_id,
            'image_id': int(frame) - 1,
        })
        annotation_id += 1

    new_path = path.split('.')[0] + '.json'
    util.file_handler.write_json(new_path, dataset)


if __name__ == '__main__':
    for video in range(1, 5):
        convert(f'HT21/train/HT21-{video:02d}/gt/gt.txt')

    for video in range(11, 16):
        convert(f'HT21/test/HT21-{video:02d}/det/det.txt')
