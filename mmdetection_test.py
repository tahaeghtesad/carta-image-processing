import logging
import sys

import cv2
import json
import numpy as np
from model_configs import configs

from detector import Detector
from util.video_handler import VideoHandler


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if __name__ == '__main__':

    img = cv2.imread('mmdetection/demo/demo.jpg')

    logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)
    config = configs['seasaw']['r_50_fpn']
    # config = configs['yolof']['r_50_c5']

    detector = Detector(config['config'], config['checkpoint'], config['person_class'])
    print(detector.model.CLASSES)
    for i in range(len(detector.model.CLASSES)):
        if detector.model.CLASSES[i] == 'person':
            print(i)
            break
    result = detector.infer(VideoHandler.extract_panes(img), detection_threshold=0.5)

