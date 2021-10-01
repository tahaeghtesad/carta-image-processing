import urllib.request
from util.file_handler import load_json, write_json


config = load_json('configs.json')

for model in config.keys():
    for variant in config[model].keys():
        url = config[model][variant]['url']
        file_name = url.split('/')[-1]

        urllib.request.urlretrieve(url, f'checkpoints/{file_name}')


        config[model][variant]['checkpoint'] = f'checkpoints/{file_name}'

        print(f'{model}|{variant}|{url}')

write_json('configs.json', config, indent=4)