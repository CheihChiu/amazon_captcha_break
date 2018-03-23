# Author : CheihChiu
# Date   : 2017-06-06

import flask
import importlib

from .util import config

def import_resources():
    from . import resources
    for mdl in resources.__all__:
        mdl_name = '.resources.' + mdl
        importlib.import_module(mdl_name, __name__)

app = flask.Flask(__name__, static_url_path='')
import_resources()
