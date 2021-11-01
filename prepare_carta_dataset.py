import util.file_handler
import cv2
import os

from util.video_handler import VideoHandler


def load_video(path):
    video_in = cv2.VideoCapture(path)
    success = True
    while success:
        success, frame = video_in.read()
        yield frame


gt_base_path = '/scratch/data/CARTA_DS2/gts/annotations_with_new_ids/'

ground_truth = [{}] * 26

print('Loading and indexing ground truth')
for i in range(1, 25):
    ground_truth[i] = util.file_handler.load_json(gt_base_path + f'video_{i}_gt.json')
    for image in ground_truth[i]['images']:
        image['annotations'] = []

    for annotation in ground_truth[i]['annotations']:
        ground_truth[i]['images'][annotation['image_id'] - 1]['annotations'].append(annotation)

dataset = {
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

base_path = 'dataset/carta'

if not os.path.isdir(base_path):
    os.makedirs(base_path, exist_ok=True)

annotation_index = 1
image_index = 0

for i in range(1, 25):
    for frame_number, frame in enumerate(load_video(f'dataset/videos/{VideoHandler.get_file_name_by_id(i)}')):
        pane_3 = VideoHandler.extract_panes(frame, 4)[3]
        dataset['images'].append({
                    'id': image_index,
                    'file_name': f'{image_index}.jpg',
                    'width': pane_3.shape[1],
                    'height': pane_3.shape[0]
                })
        cv2.imwrite(f'dataset/carta/{image_index}.jpg', pane_3)

        for annotation in ground_truth[i]['images'][frame_number if i % 2 == 1 else frame_number // 4]['annotations']:
            annotation['id'] = annotation_index
            annotation['image_id'] = image_index
            annotation['bbox'][0] -= 1920
            annotation['bbox'][1] -= 1080
            dataset['annotations'].append(annotation)

            annotation_index += 1

        image_index += 1

    util.file_handler.write_json(f'dataset/carta/annotations.json', dataset)
