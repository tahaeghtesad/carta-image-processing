import numpy as np
import requests
from tqdm import tqdm
import cv2

from util.file_handler import write_json

cvat = 'http://localhost:8080/api/v1'
token = '6f7deb746dc05b3199222ad3cc7e2c3fd4faff69'

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
target = 'dataset/image_dataset/'

project = \
requests.get(f'{cvat}/projects', headers=dict(Authorization=f'Token {token}'), params=dict(name='carta_image')).json()[
    'results'][0]

tasks = project['tasks']
image_index = 0
annotation_index = 1

images = np.zeros((len(tasks), 1080, 1920, 3))

for task in tqdm(tasks):
    task_id = task['id']
    name = task['name']

    if name == 'carta_image_ds':
        print('skipped')
        continue

    # data = requests.get(f'{cvat}/tasks/{task_id}/data', headers=dict(Authorization=f'Token {token}'), params=dict(number=0, quality='original', type='chunk'), stream=True)
    # with open(f'{target}/{name}.jpg', 'wb') as f:
    #     for chunk in data.iter_content(chunk_size=128):
    #         f.write(chunk)

    annotations = requests.get(f'{cvat}/tasks/{task_id}/annotations', headers=dict(Authorization=f'Token {token}')).json()['shapes']
    if len(annotations) == 0:
        print(f'{name} has no annotations')
        continue

    dataset['images'].append({
        'file_name': f'{name}.jpg',
        'height': 1080,
        'width': 1920,
        'id': image_index
    })

    img = cv2.imread(f'{target}/{name}.jpg')
    images[image_index] = img

    for annotation in annotations:
        points = annotation['points']
        attributes = annotation['attributes']

        x, y, w, h = points[0], points[1], points[2] - points[0], points[3] - points[1]

        dataset['annotations'].append({
            'area': w * h,
            'bbox': [x, y, w, h],
            'category_id': 1,
            'iscrowd': 0,
            'id': annotation_index,
            'image_id': image_index,
        })

        annotation_index += 1

    image_index += 1

write_json(f'{target}/annotations.coco.json', dataset)
print(images.mean(axis=(0, 1, 2)))
print(images.std(axis=(0, 1, 2)))
