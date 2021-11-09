import util.file_handler
import cv2
import random

base_path = 'dataset/carta/'
annotation_coco = util.file_handler.load_json(base_path + 'annotations.json')

for image in annotation_coco['images']:
    image['annotations'] = []

for annotation in annotation_coco['annotations']:
    annotation_coco['images'][annotation['image_id'] - 1]['annotations'].append(annotation)

for i in range(100):
    image_number_to_display = random.randrange(1, len(annotation_coco['images']))

    img = cv2.imread(base_path + annotation_coco['images'][image_number_to_display]['file_name'])
    for annotation in annotation_coco['images'][image_number_to_display]['annotations']:
        img = cv2.rectangle(img,
                            pt1=(int(annotation['bbox'][0]), int(annotation['bbox'][1])),
                            pt2=(int(annotation['bbox'][0] + annotation['bbox'][2]),
                                 int(annotation['bbox'][1] + annotation['bbox'][3])),
                            color=(255, 0, 0),
                            thickness=2)
    cv2.imshow(f'Image', img)
    cv2.waitKey()
    cv2.destroyWindow('Image')
