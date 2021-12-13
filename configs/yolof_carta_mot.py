_base_ = ['../mmdetection/configs/yolof/yolof_r50_c5_8x8_1x_coco.py']
data_root = 'dataset'

dataset_type = 'CocoDataset'

classes = ('head',)

model = dict(
    type='YOLOF',
    bbox_head=dict(
        type='YOLOFHead',
        num_classes=len(classes))
)

runner = dict(
    max_epochs=24  # From base 12
)

#img_norm_cfg = dict(
#    mean=[107.20933237, 112.40567983, 118.23271452], std=[1.0, 1.0, 1.0], to_rgb=False)

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1920, 1080), keep_ratio=True),
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
        img_scale=(1920, 1080),
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

train_dataset = [dict(
    type=dataset_type,
    ann_file=f'{data_root}/HT21/train/HT21-{i:02d}/gt/gt.json',
    img_prefix=f'{data_root}/HT21/train/HT21-{i:02d}/img1/',
    classes=classes,
    pipeline=train_pipeline
) for i in [1, 2, 3, 4]] + [dict(
        type='CocoDataset',
        ann_file=f'{data_root}/carta/train.json',
        img_prefix=f'{data_root}/carta/',
        classes=classes,
        pipeline=train_pipeline
    )]

test_dataset = [dict(
    type=dataset_type,
    ann_file=f'{data_root}/HT21/test/HT21-{i:02d}/det/det.json',
    img_prefix=f'{data_root}/HT21/test/HT21-{i:02d}/img1/',
    classes=classes,
    pipeline=test_pipeline
) for i in [11, 12, 13, 14, 15]] + [dict(
        type='CocoDataset',
        ann_file=f'{data_root}/carta/test.json',
        img_prefix=f'{data_root}/carta/',
        classes=classes,
        pipeline=train_pipeline
    )]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=1,
    train=train_dataset,
    val=test_dataset,
    test=test_dataset
)

load_from = 'checkpoints/yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth'
