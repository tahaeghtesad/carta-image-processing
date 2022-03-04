import time

import util.file_handler
import cv2
import random


def draw_box(img, bbox):
    img = cv2.rectangle(img,
                        pt1=(int(bbox[0]), int(bbox[1])),
                        pt2=(int(bbox[0] + bbox[2]),
                             int(bbox[1] + bbox[3])),
                        color=(255, 0, 0),
                        thickness=2)
    return img


base_path = 'C:\\Users\\Taha\\PycharmProjects\\carta-image-processing\\dataset\\carta_select\\'
annotation_coco = util.file_handler.load_json(base_path + 'annotations.json')

# base_path = 'dataset/carta/'
# annotation_coco = util.file_handler.load_json(base_path + 'annotations.json')

dataset = {}

for image in annotation_coco['images']:
    dataset[image['id']] = image
    dataset[image['id']]['annotations'] = []

for annotation in annotation_coco['annotations']:
    dataset[annotation['image_id']]['annotations'].append(annotation)

# for i in range(100):
#     image_number_to_display = random.randrange(0, len(annotation_coco['images']))
#
#     path = base_path + annotation_coco['images'][image_number_to_display]['file_name']
#     print(path)
#     img = cv2.imread(path)
#     for annotation in annotation_coco['images'][image_number_to_display]['annotations']:
#         if annotation['attributes']['score'] > 0.5:
        # img = draw_box(img, annotation['bbox'])
    # cv2.imshow(f'Image', img)
    # cv2.waitKey()
    # cv2.destroyWindow('Image')

for image_id, image in dataset.items():
    path = base_path + image['file_name']

    img = cv2.imread(path)
    for annotation in image['annotations']:
        img = draw_box(img, annotation['bbox'])

    cv2.imshow(f'{image["file_name"]}', img)
    cv2.waitKey()
    cv2.destroyWindow(f'{image["file_name"]}' )