import logging
import os
import sys

import cv2
from tqdm import tqdm

from detectors.mmdetectiondetector import MMDetectionDetector
from util.video_handler import VideoHandler


def infer_video(video_id, detection_threshold, detector):
    file = VideoHandler.get_file_name_by_id(video_id)
    video_in = cv2.VideoCapture(f'dataset/videos/{file}.avi')

    success, image = video_in.read()
    dim = image.shape
    if not os.path.isdir(f'dataset/annotated/'):
        os.mkdir(f'dataset/annotated')
    video_out = cv2.VideoWriter(f'dataset/annotated/{video_id}_{detector["model"]}_{detector["variant"]}_{detector["engine"].detection_class}.mp4',
                                cv2.VideoWriter_fourcc(*'MPEG'),
                                video_in.get(cv2.CAP_PROP_FPS),
                                (dim[1], dim[0])
                                # (1920, 1080)
                                )

    count = 0
    pbar = tqdm(total=int(video_in.get(cv2.CAP_PROP_FRAME_COUNT)))
    try:
        while success:
            panes = VideoHandler.extract_panes(image)
            annotations = detector['engine'].infer(panes, detection_threshold)
            for i in range(len(panes)):
                for person, score in annotations[i]:
                    panes[i] = cv2.rectangle(panes[i],
                                             pt1=(int(person[0]), int(person[1])),
                                             pt2=(int(person[2]), int(person[3])),
                                             color=detector['color'],
                                             thickness=2)
                    # print(person, score)
            video_out.write(VideoHandler.merge_panes(panes))
            # video_out.write(panes[0])

            pbar.update(1)
            count += 1

            success, image = video_in.read()

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
    detector = {
        'model': 'yolof',
        'variant': 'carta',
        'engine': MMDetectionDetector(
            # 'PedestrianDetection-NohNMS/configs/CrowdHuman/faster_rcnn_R_50_FPN_baseline_iou_0.5_noh_nms.yaml',
            # 'checkpoints/noh_nms_model_final.pth',
            'configs/yolof_carta.py',
            'work_dirs/yolof_carta/epoch_12.pth',
            'head'
        ),
        'color': (255, 0, 0)
    }
    for i in range(1, 25):
        infer_video(i, 0.5, detector)
