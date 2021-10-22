import logging
import multiprocessing
import os
import sys

import cv2
from tqdm import tqdm

from detectors.mmdetectiondetector import MMDetectionDetector
import util.file_handler
from detectors.detectron2detector import Detectron2Detector


class CocoExporter:

    def __init__(self, detector, detection_threshold) -> None:
        super().__init__()
        self.detector = detector
        self.detection_threshold = detection_threshold

    def infer_dataset(self, base_path, dataset):
        count = 0
        for image in tqdm(dataset['images']):
            video_id = image["file_name"].split('/')[0]
            # pane_id = image["file_name"].split('/')[1]
            # frame_id = image["file_name"].split('/')[2].split('.')[0]
            frame_id = image['file_name'].split('/')[1].split('.')[0]

            img = cv2.imread(f'{base_path}/{image["file_name"]}')
            people = self.detector['engine'].infer(img, self.detection_threshold)
            for person in people:
                img = cv2.rectangle(img,
                                    pt1=(int(person[0][0]), int(person[0][1])),
                                    pt2=(int(person[0][2]), int(person[0][3])),
                                    color=self.detector['color'],
                                    thickness=2)
                dataset['annotations'].append({
                    'id': count,
                    'image_id': image['id'],
                    'category_id': 1,
                    'segmentation': [],
                    'attributes': {
                        'score': float(person[1]),
                    },
                    'bbox': [
                        float(person[0][0]),  # x
                        float(person[0][1]),  # y
                        float(person[0][2] - person[0][0]),  # width
                        float(person[0][3] - person[0][1])  # height
                    ]
                })
                count += 1
            write_back_path = f'{base_path}/{video_id}_annotated/{self.detector["model"]}/{self.detector["variant"]}/'
            if not os.path.isdir(write_back_path):
                os.makedirs(write_back_path, exist_ok=True)
            cv2.imwrite(f'{write_back_path}/{frame_id}.jpg', img)

        return dataset


def get_detector(framework):
    if framework == 'detectron2':
        return Detectron2Detector
    elif framework == 'mmdetection':
        return MMDetectionDetector


def run_inference_on_video(video_id, configs, framework):
    dataset = util.file_handler.load_json(f'{base_path}/annotations/video_{video_id}.coco.json')

    for model in configs.keys():
        for variant in configs[model].keys():
            logging.getLogger(__name__).info(f'Loading model {model}/{variant}.')
            logging.getLogger(__name__).info(f'config path: {configs[model][variant]["config"]}')
            logging.getLogger(__name__).info(f'checkpoint path: {configs[model][variant]["checkpoint"]}-{type(configs[model][variant]["checkpoint"])}')

            detector = {
                'model': model,
                'variant': variant,
                'engine': get_detector(framework)(configs[model][variant]['config'], configs[model][variant]['checkpoint'], 'person'),
                'color': (255, 0, 0)
            }

            exporter = CocoExporter(detector, 0.5)

            logger.info(f'Running export for model "{detector["model"]}" with variant "{detector["variant"]}" on video "video_{video_id}"...')
            new_dataset = exporter.infer_dataset(base_path, dataset)
            util.file_handler.write_json(
                f'{base_path}/annotations/video_{video_id}_{detector["model"]}_{detector["variant"]}.coco.json',
                new_dataset)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    base_path = 'dataset/split_pane'

    # for video_id in range(1, 25):
    #     run_inference_on_video(video_id)

    run_inference_on_video(25, util.file_handler.load_json('detectron2_configs.json'), 'detectron2')
    run_inference_on_video(25, util.file_handler.load_json('configs.json'), 'mmdetection')

