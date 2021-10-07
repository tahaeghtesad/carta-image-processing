_base_ = ['mmdetection/configs/yolof/yolof_r50_c5_8x8_1x_coco.py']
data_root = 'crowdhuman/'
classes = ('person', 'head', 'body')

model = dict(
    roi_head=dict(
        type='YOLOFHead',
        bbox_head=dict(num_classes=len(classes)))
)

dataset_type = 'COCODataset'

data = dict(
    train=dict(
        ann_file=data_root + 'annotation_train.json',
        img_prefix=data_root + 'images_train/',
        classes=classes,
        type=dataset_type
    ),
    val=dict(
        ann_file=data_root + 'annotation_val.json',
        img_prefix=data_root + 'images_val/',
        classes=classes,
        type=dataset_type
    ),
    test=None,
)

load_from = 'checkpoints/yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth"'
