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


configs = {
    'yolof': {
        'r_50_c5': {
            'config': 'mmdetection/configs/yolof/yolof_r50_c5_8x8_1x_coco.py',
            'checkpoint': 'checkpoints/yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth',
            'person_class': 0,
            'color': (0, 0, 255)
        }
    },
    'seasaw': {
        'r_50_fpn': {
            'config': 'mmdetection/configs/seesaw_loss/mask_rcnn_r50_fpn_sample1e-3_seesaw_loss_normed_mask_mstrain_2x_lvis_v1.py',
            'checkpoint': 'checkpoints/mask_rcnn_r50_fpn_sample1e-3_seesaw_loss_normed_mask_mstrain_2x_lvis_v1-cd0f6a12.pth',
            'person_class': 0,
            'color': (0, 255, 0)
        }
    },
    'yolox': {
        'r_50_fpn': {
            'config': 'mmdetection/configs/yolox/yolox_tiny_8x8_300e_coco.py',
            'checkpoint': 'checkpoints/yolox_tiny_8x8_300e_coco_20210806_234250-4ff3b67e.pth',
            'person_class': 0,
            'color': (255, 0, 0)
        }
    }
}
if __name__ == '__main__':

    img = 'mmdetection/demo/demo.jpg'

    logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    for model in configs.keys():
        for variant in configs[model].keys():
            detector = Detector(**configs[model][variant])
            result = inference_detector(detector.model, img)
            show_result_pyplot(detector.model, img, result, score_thr=0.3)