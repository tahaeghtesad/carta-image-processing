_base_ = ['../mmdetection/configs/yolof/yolof_r50_c5_8x8_1x_coco.py']
data_root = 'dataset/'
crowdhuman_data_root = data_root + 'crowdhuman/'

classes = ('person',)

model = dict(
    type='YOLOF',
    bbox_head=dict(
        type='YOLOFHead',
        num_classes=len(classes))
)

dataset_type = 'CocoDataset'

runner = dict(
    max_epochs=48  # From base 12
)

#img_norm_cfg = dict(
#    mean=[107.20933237, 112.40567983, 118.23271452], std=[1.0, 1.0, 1.0], to_rgb=False)

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

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
    samples_per_gpu=12,
    workers_per_gpu=1,
    train=dict(
        ann_file=[crowdhuman_data_root + 'annotation_train.json'],
        img_prefix=[crowdhuman_data_root + 'images_train/'],
        classes=classes,
        type=dataset_type,
        pipeline=train_pipeline
    ),
    val=dict(
        ann_file=[crowdhuman_data_root + 'annotation_val.json'],
        img_prefix=[crowdhuman_data_root + 'images_val/'],
        classes=classes,
        type=dataset_type,
        pipeline=test_pipeline
    ),
    test=dict(
        ann_file=[crowdhuman_data_root + 'annotation_val.json'],
        img_prefix=[crowdhuman_data_root + 'images_val/'],
        classes=classes,
        type=dataset_type,
        pipeline=test_pipeline
    )
)

load_from = 'checkpoints/yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth'
