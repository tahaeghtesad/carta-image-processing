import logging
import os
import sys

import cv2
from tqdm import tqdm

import util.file_handler
from detectors.mmdetectiondetector import MMDetectionDetector
from util.video_handler import VideoHandler


def infer_video(path, file, detection_threshold, detector):
    video_in = cv2.VideoCapture(f'{path}/{file}')

    success, image = video_in.read()
    assert success, f'Could not read video file {path}/{file}'

    dataset = {
        'info': {},
        'licenses': [],
        'images': [],
        'annotations': [],
        'categories': [
            {
                'id': 1,
                'supercategory': 'none',
                'name': 'head'
            }
        ]
    }

    dim = image.shape
    if not os.path.isdir(f'dataset/annotated/'):
        os.mkdir(f'dataset/annotated')

    split_dest = f'{path}/{file.split(".")[0]}/'
    if not os.path.isdir(split_dest):
        os.mkdir(split_dest)

    video_out = cv2.VideoWriter(f'dataset/annotated/{file.split(".")[0]}_{detector["model"]}_{detector["variant"]}_{detector["engine"].detection_class}.mp4',
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                video_in.get(cv2.CAP_PROP_FPS),
                                (dim[1], dim[0])
                                )

    count = 0
    image_id = 1
    annotation_id = 1
    pbar = tqdm(total=int(video_in.get(cv2.CAP_PROP_FRAME_COUNT)))
    try:
        while success:

            cv2.imwrite(f'{split_dest}/{image_id:06d}.jpg', image)

            panes = [image]

            dataset['images'].append({
                'file_name': f'{image_id:06d}.jpg',
                'height': 1080,
                'width': 1920,
                'id': image_id
            })

            annotations = detector['engine'].infer(panes, detection_threshold)
            for i in range(len(panes)):
                for person, score in annotations[i]:

                    dataset['annotations'].append({
                        'id': annotation_id,
                        'image_id': image_id,
                        'category_id': 1,
                        'segmentation': [],
                        'attributes': {
                            'score': float(score),
                        },
                        'bbox': [
                            float(person[0]),  # x
                            float(person[1]),  # y
                            float(person[2] - person[0]),  # width
                            float(person[3] - person[1])  # height
                        ]
                    })

                    panes[i] = cv2.rectangle(panes[i],
                                             pt1=(int(person[0]), int(person[1])),
                                             pt2=(int(person[2]), int(person[3])),
                                             color=detector['color'],
                                             thickness=2)
                    annotations += 1
                    # print(person, score)
            # video_out.write(VideoHandler.merge_panes(panes))
            video_out.write(panes[0])

            pbar.update(1)
            count += 1
            image_id += 1

            success, image = video_in.read()

            util.file_handler.write_json(f'{split_dest}/dataset.json', dataset)

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
        # 'model': 'yolof',
        # 'variant': 'carta',
        'model': 'faster-rcnn',
        'variant': 'cartai_full',
        'engine': MMDetectionDetector(
            # 'PedestrianDetection-NohNMS/configs/CrowdHuman/faster_rcnn_R_50_FPN_baseline_iou_0.5_noh_nms.yaml',
            # 'checkpoints/noh_nms_model_final.pth',
            # 'configs/yolof_carta.py',
            # 'work_dirs/yolof_carta/epoch_12.pth',
            'configs/faster_rcnn_cartai_full.py',
            'work_dirs/faster_rcnn_cartai_full/epoch_24.pth',
            'head'
        ),
        'color': (255, 0, 0)
    }

    #infer_video('test/Bus 505-Video-01-16-2021 19-33-23_01.avi', 0.3, detector)
    #infer_video('Bus 124-Video-07-23-2021 15-41-05.avi', 0.3, detector)
    #infer_video('Bus 137-Video-07-23-2021 16-40-55.avi', 0.3, detector)
    # infer_video('dataset/videos/test/', 'Bus 504-Video-01-20-2021 11-31-32_01.avi', 0.1, detector)
    # infer_video('dataset/videos/test/', 'Bus 505-Video-01-16-2021 19-33-23_01.avi', 0.1, detector)
    infer_video('dataset/new_video/', 'Bus 154-Video-02-28-2022 18-45-18.avi', 0.0, detector)
