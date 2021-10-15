_base_ = ['new_yolof_config.py']

optimizer = dict(_delete_=True, type='Adam', lr=0.0003, weight_decay=0.0001)

load_from = 'checkpoints/yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth'
