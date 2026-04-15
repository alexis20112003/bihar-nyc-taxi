import os
import yaml

# project root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# full config path
CONFIG_PATH = os.path.join(ROOT_DIR, 'config.yml')

with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.load(f, Loader=yaml.SafeLoader)