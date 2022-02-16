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


base_path = 'dataset/image_dataset/'
annotation_coco = util.file_handler.load_json(base_path + 'carta_image_dataset.json')

# base_path = 'dataset/carta/'
# annotation_coco = util.file_handler.load_json(base_path + 'annotations.json')

for image in annotation_coco['images']:
    image['annotations'] = []

for annotation in annotation_coco['annotations']:
    annotation_coco['images'][annotation['image_id']]['annotations'].append(annotation)

for i in range(100):
    image_number_to_display = random.randrange(0, len(annotation_coco['images']))

    path = base_path + annotation_coco['images'][image_number_to_display]['file_name']
    print(path)
    img = cv2.imread(path)
    for annotation in annotation_coco['images'][image_number_to_display]['annotations']:
        # if annotation['attributes']['score'] > 0.5:
        img = draw_box(img, annotation['bbox'])
    cv2.imshow(f'Image', img)
    cv2.waitKey()
    cv2.destroyWindow('Image')
