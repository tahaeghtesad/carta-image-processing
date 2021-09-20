import logging
import os
import sys
from os import listdir
from os.path import isfile, join
import cv2
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
import numpy as np

from mmdet.apis import show_result_pyplot

from mmdetection_test import configs, NumpyEncoder

from detector import Detector
import json

from video_handler import VideoHandler

if __name__ == '__main__':

    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    video_id = sys.argv[2]
    detection_threshold = float(sys.argv[3])

    detectors = []
    for model in configs.keys():
        for variant in configs[model].keys():
            detectors.append({
                'model': model,
                'variant': variant,
                'engine': Detector(configs[model][variant]['config'],
                                   configs[model][variant]['checkpoint'],
                                   configs[model][variant]['person_class'],
                                   ),
                'color': configs[model][variant]['color']
            })

    file = VideoHandler.get_file_name_by_id(video_id)
    video_in = cv2.VideoCapture(f'dataset/videos/{file}.avi')

    success, image = video_in.read()
    dim = image.shape
    if not os.path.isdir(f'dataset/annotated/'):
        os.mkdir(f'dataset/annotated')
    video_out = cv2.VideoWriter(f'dataset/annotated/{video_id}.avi',
                                cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                video_in.get(cv2.CAP_PROP_FPS),
                                (dim[1], dim[0]))

    count = 0
    pbar = tqdm(total=int(video_in.get(cv2.CAP_PROP_FRAME_COUNT)))
    while success:
        panes = VideoHandler.extract_panes(image)
        annotated_panes = []
        for detector in detectors:
            annotations = detector['engine'].infer(panes, detection_threshold)
            for i in range(len(panes)):
                for person in annotations[i]:
                    panes[i] = cv2.rectangle(panes[i],
                                             pt1=(int(person[0]), int(person[1])),
                                             pt2=(int(person[2]), int(person[3])),
                                             color=detector['color'],
                                             thickness=2)
            video_out.write(VideoHandler.merge_panes(panes))
            success, image = video_in.read()
        pbar.update(1)

    video_in.release()
    video_out.release()
    pbar.close()
