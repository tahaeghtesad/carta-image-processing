_base_ = ['../mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_iou_1x_coco.py']
data_root = 'dataset/'
crowdhuman_data_root = data_root + 'crowdhuman/'
carta_data_root = data_root + 'carta/'

classes = ('head', 'person', 'body')

model = dict(
    type='FasterRCNN',
    roi_head=dict(
        bbox_head=dict(
            num_classes=len(classes)
        )
    )
)

dataset_type = 'CocoDataset'

runner = dict(
    max_epochs=12  # From base 12
)

img_norm_cfg = dict(
    mean=[107.20933237, 112.40567983, 118.23271452], std=[1.0, 1.0, 1.0], to_rgb=False)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomShift', shift_ratio=0.5, max_shift_px=32),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    workers_per_gpu=1,
    train=dict(
        ann_file=[carta_data_root + 'train.json', crowdhuman_data_root + 'annotation_train.json'],
        img_prefix=[carta_data_root, crowdhuman_data_root + 'images/' + 'train/'],
        classes=classes,
        type=dataset_type
    ),
    val=dict(
        ann_file=[carta_data_root + 'test.json'],
        img_prefix=[carta_data_root],
        classes=classes,
        type=dataset_type
    ),
    test=dict(
        ann_file=[carta_data_root + 'test.json'],
        img_prefix=[carta_data_root],
        classes=classes,
        type=dataset_type
    )
)
