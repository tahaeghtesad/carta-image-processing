_base_ = ['new_yolof_config.py']

optimizer = dict(_delete_=True, type='Adam', lr=0.0003, weight_decay=0.0001)
