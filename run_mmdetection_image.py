import logging
import os
import sys

import cv2
from tqdm import tqdm

from detectors.mmdetectiondetector import MMDetectionDetector
from util.video_handler import VideoHandler
from util.file_handler import load_json, write_json


def infer_image(coco_file, detection_threshold, detector):
    coco_data = '/'.join(coco_file.split('/')[:-1])
    logger.info(f'coco_data: {coco_data}')
    coco_file_name = coco_file.split('/')[-1].split('.')[0]
    logger.info(f'coco_file_name: {coco_file_name}')
    dataset = load_json(coco_file)
    if not os.path.isdir(f'dataset/annotated/image/'):
        os.makedirs(f'dataset/annotated/image')

    if not os.path.isdir(f'dataset/annotated/image/{coco_file_name}.{detector["name"]}/'):
        os.mkdir(f'dataset/annotated/image/{coco_file_name}.{detector["name"]}/')

    pbar = tqdm(dataset['images'])
    for image in pbar:
        pbar.set_description(f'{image["file_name"]}')
        img = cv2.imread(coco_data + '/' + image["file_name"])
        annotations = detector['engine'].infer_image([img], detection_threshold)
        for person, score in annotations[0]:
            img = cv2.rectangle(img,
                                pt1=(int(person[0]), int(person[1])),
                                pt2=(int(person[2]), int(person[3])),
                                color=detector['color'],
                                thickness=2)
        cv2.imwrite(f'dataset/annotated/image/{coco_file_name}.{detector["name"]}/{image["file_name"]}', img)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    detector = {
        'name': 'faster-rcnn_10',
        'engine': MMDetectionDetector(
            'configs/faster_rcnn_cartai.py',
            'work_dirs/faster_rcnn_cartai/epoch_10.pth',
            'head'
        ),
        'color': (0, 255, 0)
    }

