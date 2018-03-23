# Author : CheihChiu
# Date   : 2017-06-06

import yaml
from . import logger

options = yaml.load(open('application/conf/conf.yaml'))
logger.setup_logging('application/conf/logger.yaml')

