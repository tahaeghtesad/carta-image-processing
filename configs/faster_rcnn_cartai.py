_base_ = ['../mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_iou_1x_coco.py']
data_root = 'dataset/image_dataset/'

classes = ('head',)

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
    max_epochs=24  # From base 12
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
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
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        ann_file=data_root + 'train.json',
        img_prefix=data_root,
        classes=classes,
        type=dataset_type,
        pipeline=train_pipeline
    ),
    val=dict(
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        classes=classes,
        type=dataset_type,
        pipeline=test_pipeline
    ),
    test=dict(
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        classes=classes,
        type=dataset_type,
        pipeline=test_pipeline
    )
)

# load_from = 'checkpoints/dh_faster_rcnn_r50_fpn_1x_coco_20200130-586b67df.pth'