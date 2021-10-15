import subprocess
import time


def run(target):
    subprocess.run(['sbatch', 'train_on_crowdhuman.sh', target])


targets = [
    'new_yolof_config.py',
    'yolof_adam.py',
    'yolof_adam_loaded.py',
    'yolof_sgd_0.2.py',
    'yolof_sgd_0.12.py',
    'yolof_sgd_0.12_loaded.py'
]

if __name__ == '__main__':
    for target in targets:
        print(f'running training for {target}')
        run(target)
        time.sleep(60)
