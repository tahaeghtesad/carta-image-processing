import logging
import sys

import cv2
import json
import numpy as np

from detector import Detector
from file_handler import load_json
from util.video_handler import VideoHandler


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if __name__ == '__main__':

    img = cv2.imread('mmdetection/demo/demo.jpg')
    configs = load_json('configs.json')

    logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)
    config = configs['scnet']['r-101-fpn']

    detector = Detector(config['config'], config['checkpoint'])
    print(detector.model.CLASSES)
    for i in range(len(detector.model.CLASSES)):
        if detector.model.CLASSES[i] == 'person':
            print(i)
            break
    result = detector.infer(VideoHandler.extract_panes(img), detection_threshold=0.0)
    print(result)

