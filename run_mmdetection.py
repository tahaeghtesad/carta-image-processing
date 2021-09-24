import logging
import os
import sys

import cv2
from tqdm import tqdm

from detector import Detector
from model_configs import configs
from util.video_handler import VideoHandler


def infer_video(video_id, detection_threshold, detector):
    file = VideoHandler.get_file_name_by_id(video_id)
    video_in = cv2.VideoCapture(f'dataset/videos/{file}.avi')

    success, image = video_in.read()
    dim = image.shape
    if not os.path.isdir(f'dataset/annotated/'):
        os.mkdir(f'dataset/annotated')
    video_out = cv2.VideoWriter(f'dataset/annotated/{video_id}_{detector["model"]}_{detector}.avi',
                                cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                video_in.get(cv2.CAP_PROP_FPS),
                                (dim[1], dim[0]))

    count = 0
    pbar = tqdm(total=int(video_in.get(cv2.CAP_PROP_FRAME_COUNT)))
    try:
        while success:
            panes = VideoHandler.extract_panes(image)
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
        count += 1

    except Exception as e:
        logger.error(f'Task failed! Model: "{detector["model"]}" Variant: "{detector["variant"]}" Frame: {count}')
        logger.exception(e)

    video_in.release()
    video_out.release()
    pbar.close()


if __name__ == '__main__':

    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    for model in configs.keys():
        for variant in configs[model].keys():
            detector = {
                        'model': model,
                        'variant': variant,
                        'engine': Detector(configs[model][variant]['config'],
                                           configs[model][variant]['checkpoint'],
                                           ),
                        'color': configs[model][variant]['color']
                    }
            infer_video('VID001', 0.5, detector)