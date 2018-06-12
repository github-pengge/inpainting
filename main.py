from networks import create_Generator
from data import CelebA 
from model import InpaintingModel
import json, time, os 
from config import config


G = create_Generator(config)
celeba = CelebA(os.path.expanduser('~/datasets/celeba-hq-1024x1024.h5'), os.path.expanduser('~/datasets/holes_hq.hdf5'))
config['model_dir'] = config['model_dir'].replace('<time>', time.strftime("%Y-%b-%d %H_%M_%S"))
os.makedirs(config['model_dir'])
with open(os.path.join(config['model_dir'], 'config.json'), 'w') as f:
	json.dump(config, f)

model = InpaintingModel(G, celeba, config)
model.run()
