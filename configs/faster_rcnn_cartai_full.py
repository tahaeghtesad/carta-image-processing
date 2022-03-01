_base_ = ['../mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_iou_1x_coco.py']

# data_root = 'dataset/image_dataset/'
new_root = 'dataset/image_dataset/'
old_root = 'dataset/carta_select/'

classes = ('head',)

model = dict(
    type='FasterRCNN',
    roi_head=dict(
        bbox_head=dict(
            num_classes=len(classes)
        )
    )
)

optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)

dataset_type = 'CocoDataset'

runner = dict(
    max_epochs=24  # From base 12
)

img_norm_cfg = dict(
    mean=[101.8627779, 98.64721287, 99.20499043],
    std=[71.98746042, 74.29544418, 74.45167525],
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomShift', shift_ratio=0.5, max_shift_px=32),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Corrupt', corruption='gaussian_noise', severity=1), #Gausian noise makes the loss to become 'nan'
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
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
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=[dict(
        ann_file=new_root + 'train.json',
        img_prefix=new_root,
        classes=classes,
        type=dataset_type,
        pipeline=train_pipeline
    ), dict(
        ann_file=old_root + 'train.json',
        img_prefix=old_root,
        classes=classes,
        type=dataset_type,
        pipeline=train_pipeline
    )],
    val=[dict(
        ann_file=new_root + 'test.json',
        img_prefix=new_root,
        classes=classes,
        type=dataset_type,
        pipeline=train_pipeline
    ), dict(
        ann_file=old_root + 'test.json',
        img_prefix=old_root,
        classes=classes,
        type=dataset_type,
        pipeline=train_pipeline
    )],
    test=[dict(
        ann_file=new_root + 'test.json',
        img_prefix=new_root,
        classes=classes,
        type=dataset_type,
        pipeline=train_pipeline
    ), dict(
        ann_file=old_root + 'test.json',
        img_prefix=old_root,
        classes=classes,
        type=dataset_type,
        pipeline=train_pipeline
    )]
)

# load_from = 'checkpoints/dh_faster_rcnn_r50_fpn_1x_coco_20200130-586b67df.pth'