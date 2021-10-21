import math
from collections import defaultdict

import util.file_handler
from detectors.mmdetectiondetector import MMDetectionDetector
from util.video_handler import VideoHandler
import csv

from util.file_handler import load_json

from tqdm import tqdm

configs = util.file_handler.load_json('configs.json')
detectors = defaultdict(lambda: dict())
videos = {}
print(f'Loading Videos')
for video_id in tqdm(range(1, 25)):
    videos[video_id] = VideoHandler(f'dataset/videos/{VideoHandler.get_file_name_by_id(video_id)}.avi', 5)


def compare(video, model, variant):

    detector = detectors[model][variant]
    handler = videos[video]
    before_frames = [VideoHandler.extract_panes(frame)[3] for frame in handler.first_frames]
    after_frames = [VideoHandler.extract_panes(frame)[3] for frame in handler.last_frames]
    before_inference_count = [len(frame_bbox) for frame_bbox in detector.infer(before_frames, detection_threshold=0.5)]
    after_inference_count = [len(frame_bbox) for frame_bbox in detector.infer(after_frames, detection_threshold=0.5)]
    before_inference = sum(before_inference_count) / len(before_inference_count)
    after_inference = sum(after_inference_count) / len(after_inference_count)

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

    # print(
    #     f'Video:{video}|model:{model}|variance:{variance}|Before stop:{bboxes_first}/{bboxes_gt_first}|After stop: {bboxes_last}/{bboxes_gt_last}')

    return {
        'video': video,
        'model': model,
        'variant': variant,
        'before': before_inference,
        'before_gt': bboxes_gt_first,
        'after': after_inference,
        'after_gt': bboxes_gt_last,
        'pane_color': VideoHandler.get_info_row_by_id(video, 'Panel 4 (RGB/GRAY)')
    }


def analyse(result, when, loss):
    return loss(result[when], result[f'{when}_gt'])


if __name__ == '__main__':
    fd = open('analysis.csv', 'w')
    writer = csv.writer(fd)
    writer.writerow(['model', 'variant', 'loss_type', 'loss'])

    losses = {
        # 'mse': lambda x, y: (x - y) ** 2,
        # 'mae': lambda x, y: math.fabs(x - y),
        'mape': lambda x, y: math.fabs(x - y) / y * 100

    }

    for model in configs.keys():
        for variant in configs[model].keys():
            print(f'Loading model "{model}" with variant "{variant}"')
            detectors[model][variant] = MMDetectionDetector(configs[model][variant]['config'], configs[model][variant]['checkpoint'])

    for loss in losses.keys():
        for model in configs.keys():
            for variant in configs[model].keys():
                print(f'Analysing model "{model}" with variant "{variant}"')
                results = [compare(video, model, variant) for video in range(1, 25)]

                for when in ['before', 'after']:
                    loss_sum = 0
                    for result in results:
                        loss_sum += analyse(result, when, losses[loss])

                loss_avg = loss_sum / len(results) / 2

                print(f'Model: {model}\tVariant: {variant}\tloss type: {loss}\tloss: {loss_avg:.2f}')
                writer.writerow([model, variant, loss, loss_avg])
