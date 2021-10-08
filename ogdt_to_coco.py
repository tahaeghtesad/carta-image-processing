import json
import cv2
from tqdm import tqdm

import util.file_handler

mode = 'train'
print(f'Running for mode: {mode}')

with open(f'crowdhuman/annotation_{mode}.odgt') as fd:
    annotation_lines = fd.readlines()

dataset = {
    'images': [],
    'annotations': [],
    'categories': [
        {
            'id': 1,
            'supercategory': 'none',
            'name': 'person'
        },
        {
            'id': 2,
            'supercategory': 'none',
            'name': 'head'
        },
        {
            'id': 3,
            'supercategory': 'none',
            'name': 'body'
        }
    ]
}

image_id = 0
annotation_id = 0
head_sizes = []

for line in tqdm(annotation_lines):
    jsonified = json.loads(line)
    name = jsonified['ID']
    img = cv2.imread(f'crowdhuman/images_{mode}/{name}.jpg')
    dataset['images'].append({
        'file_name': f'{name}.jpg',
        'height': img.shape[0],
        'width': img.shape[1],
        'id': image_id
    })

    for box in jsonified['gtboxes']:
        if box['tag'] == 'person' and\
                ('ignore' not in box['head_attr'] or box['head_attr']['ignore'] == 0) and \
                ('occ' not in box['head_attr'] or box['head_attr']['occ'] == 0) and \
                ('unsure' not in box['head_attr'] or box['head_attr']['unsure'] == 0):
            hbox = box['hbox']  # head box # 2
            fbox = box['fbox']  # full box # 3
            vbox = box['vbox']  # visible box # 1

            # visible, marked as person
            dataset['annotations'].append({
                'area': vbox[2] * vbox[3],
                'bbox': vbox,
                'category_id': 1,
                'iscrowd': 0,
                'id': annotation_id,
                'image_id': image_id,
            })
            annotation_id += 1

            # head box
            dataset['annotations'].append({
                'area': hbox[2] * hbox[3],
                'bbox': hbox,
                'category_id': 2,
                'iscrowd': 0,
                'id': annotation_id,
                'image_id': image_id,
            })
            annotation_id += 1
            head_sizes.append(dataset['annotations'][-1]['area'])

            # Full body, marked as body
            dataset['annotations'].append({
                'area': fbox[2] * fbox[3],
                'bbox': fbox,
                'category_id': 3,
                'iscrowd': 0,
                'id': annotation_id,
                'image_id': image_id,
            })
            annotation_id += 1

    image_id += 1

print(f'Average head area: {sum(head_sizes)/len(head_sizes)}')
print(f'Total images: {image_id}')
print(f'Total annotations: {annotation_id}')
util.file_handler.write_json(f'crowdhuman/annotation_{mode}.json', dataset)
