import logging

import numpy as np
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor

from detectors.detector import Detector
import torch


class Detectron2Detector(Detector):
    def __init__(self, config, checkpoint, detection_class, device='cuda') -> None:
        super().__init__(config, checkpoint, detection_class, device)
        self.logger = logging.getLogger(__name__)
        self.predictor = DefaultPredictor(self.__setup_cfg())

    def __setup_cfg(self):
        cfg = get_cfg()
        cfg.merge_from_file(self.config)

        cfg.MODEL.DEVICE = self.device
        cfg.MODEL.WEIGHTS = self.checkpoint
        cfg.freeze()
        return cfg

    def batch_infer(self, img):
        ret = []
        for image in img:
            ret.append(self.predictor(image))
        return ret

    def infer(self, img, detection_threshold):
        assert isinstance(img, list), 'Only provide list of four panes'
        predictions = self.batch_infer(img)
        persons = [[] for _ in range(len(img))]
        for pane in range(len(img)):
            if 'instances' in predictions[pane]:
                instances = predictions[pane]['instances'].to(torch.device(self.device))

                boxes = instances.pred_boxes if instances.has("pred_boxes") else None
                scores = instances.scores if instances.has("scores") else None
                classes = instances.pred_classes.tolist() if instances.has("pred_classes") else None

                for box, score, class_ in zip(boxes, scores, classes):
                    if score > detection_threshold:
                        persons[pane].append((box, score))

        return persons
