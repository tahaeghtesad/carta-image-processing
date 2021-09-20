import logging
import sys

from mmdet.apis import init_detector, inference_detector, show_result_pyplot, get_root_logger
import json
import time
import numpy as np

from detector import Detector


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if __name__ == '__main__':

    img = 'mmdetection/demo/demo.jpg'

    logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    for model in configs.keys():
        for variant in configs[model].keys():
            detector = Detector(**configs[model][variant])
            result = inference_detector(detector.model, img)
            show_result_pyplot(detector.model, img, result, score_thr=0.3)
