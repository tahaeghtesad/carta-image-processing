import logging
import time
from typing import List
import numpy as np

from mmdet.apis import init_detector, inference_detector


class Detector:
    def __init__(self, config, checkpoint, person_class, device='cuda:0') -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__)

        start = time.time()
        # init a detector
        self.model = init_detector(config, checkpoint, device=device)
        self.person_class = person_class
        self.logger.debug(f'Took {time.time() - start:.2f} (s) to init')

    def infer(self, img, detection_threshold):
        # assert isinstance(img, str), 'Only provide image paths.'
        # assert isinstance(img, np.ndarray), 'Only provided loaded images'
        assert isinstance(img, list), 'Only provide list of four panes'
        start = time.time()
        result = inference_detector(self.model, img)
        if self.model == 'seasaw':
            result = result[0]
        self.logger.debug(f'Took {time.time() - start:.2f} (s) to infer.')

        persons = [[]] * len(img)

        for pane in range(len(img)):
            for each_person in result[pane][self.person_class]:
                if each_person[4] > detection_threshold:
                    persons[pane].append(each_person)

        return persons
