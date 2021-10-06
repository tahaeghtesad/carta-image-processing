_base_ = ['mmdetection/configs/yolof/yolof_r50_c5_8x8_1x_coco.py']
data_root = 'crowdhuman/'
data = dict(
    train=dict(
        ann_file=data_root + 'annotation_train.json',
        img_prefix=data_root + 'images_train/'
    ),
    val=dict(
        ann_file=data_root + 'annotation_val.json',
        img_prefix=data_root + 'images_val/'
    ),
    test=None,
)
