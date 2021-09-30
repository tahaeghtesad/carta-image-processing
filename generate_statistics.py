import json
import math

import cv2
from util.video_handler import VideoHandler
import csv

from model_configs import configs
from util.file_handler import load_json


def get_frame_count(path):
    video_in = cv2.VideoCapture(path)
    frame_count = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))
    assert frame_count > 1, f'Bad path {path}'
    return frame_count


def compare(video, model, variance):
    inference_set = load_json(f'dataset/split/annotations/video_{video}_{model}_{variance}.coco.json')

    ground_truth_set = load_json(
        f'dataset/annotations/{VideoHandler.get_coco_annotation_by_id(video)}/annotations/instances_default.json')

    first_annotation_image_id = ground_truth_set['annotations'][0]['image_id']
    bboxes_gt_first = len(
        [annotation for annotation in ground_truth_set['annotations'] if
         annotation['image_id'] == first_annotation_image_id])

    last_annotation_image_id = ground_truth_set['annotations'][-1]['image_id']
    bboxes_gt_last = len(
        [annotation for annotation in ground_truth_set['annotations'] if
         annotation['image_id'] == last_annotation_image_id])

    first_frame_ids = [image['id'] for image in inference_set['images']][:5]
    first_frames_list = [0] * 5
    for annotation in inference_set['annotations']:
        for i, image_id in enumerate(first_frame_ids):
            if annotation['image_id'] == image_id:
                first_frames_list[i] += 1

    bboxes_first = sum(first_frames_list) / len(first_frames_list)

    last_frame_ids = [image['id'] for image in inference_set['images']][-5:]
    last_frame_list = [0] * 5
    for annotation in inference_set['annotations']:
        for i, image_id in enumerate(last_frame_ids):
            if annotation['image_id'] == image_id:
                last_frame_list[i] += 1

    bboxes_last = sum(last_frame_list) / len(last_frame_list)

    # print(
    #     f'Video:{video}|model:{model}|variance:{variance}|Before stop:{bboxes_first}/{bboxes_gt_first}|After stop: {bboxes_last}/{bboxes_gt_last}')

    return {
        'video': video,
        'model': model,
        'variant': variant,
        'before': bboxes_first,
        'before_gt': bboxes_gt_first,
        'after': bboxes_last,
        'after_gt': bboxes_gt_last,
        'pane_color': VideoHandler.get_info_row_by_id(video, 'Panel 4 (RGB/GRAY)')
    }


def analyse(result, when, loss):
    return loss(result[when], result[f'{when}_gt'])


if __name__ == '__main__':
    fd = open('analysis.csv', 'w')
    writer = csv.writer(fd)
    writer.writerow(['model', 'variant', 'color', 'when', 'loss_type', 'loss'])

    losses = {
        'mse': lambda x, y: (x - y) ** 2,
        'mae': lambda x, y: math.fabs(x - y),
        'mape': lambda x, y: math.fabs(x - y) / y * 100

    }

    for loss in losses.keys():
        for model in configs.keys():
            for variant in configs[model].keys():
                types = {
                    'RGB': [],
                    'GRAY': []
                }
                for video in range(1, 25):
                    result = compare(video, model, variant)
                    for pane_color in types.keys():
                        if result['pane_color'] == pane_color:
                            types[pane_color].append(result)

                for color in types.keys():
                    for when in ['before', 'after']:
                        loss_avg = 0
                        for result in types[color]:
                            loss_avg += analyse(result, when, losses[loss]) / len(types[color])
                        print(f'Model: {model}\tVariant: {variant}\tColor: {color}\tWhen: {when} \tloss type: {loss}\tloss: {loss_avg:.2f}')
                        writer.writerow([model, variant, color, when, loss, loss_avg])
