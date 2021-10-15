_base_ = ['new_yolof_config.py']

optimizer = dict(
    type='SGD',
    lr=0.12,
    momentum=0.9,
    weight_decay=0.0001
)