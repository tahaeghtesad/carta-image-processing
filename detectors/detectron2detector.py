import logging

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor

from detectors.detector import Detector
import torch


class Detectron2Detector(Detector):
    def __init__(self, config, checkpoint, detection_class, device='cpu') -> None:
        super().__init__(config, checkpoint, detection_class, device)
        self.logger = logging.getLogger(__name__)
        self.predictor = DefaultPredictor(self.__setup_cfg())

    def __setup_cfg(self):
        cfg = get_cfg()
        cfg.merge_from_file(self.config)

        cfg.merge_from_list(['MODEL.WEIGHTS', f'{self.checkpoint}'])
        cfg.MODEL.DEVICE = self.device
        cfg.freeze()
        return cfg

    def infer(self, img, detection_threshold):
        predictions = self.predictor(img)
        ret = []
        if 'instances' in predictions:
            instances = predictions['instances'].to(torch.device(self.device))

            boxes = instances.pred_boxes if instances.has("pred_boxes") else None
            scores = instances.scores if instances.has("scores") else None
            classes = instances.pred_classes.tolist() if instances.has("pred_classes") else None

            for box, score, class_ in zip(boxes, scores, classes):
                if score > detection_threshold:
                    ret.append(box)

        return ret
