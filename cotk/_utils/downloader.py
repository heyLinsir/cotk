"""
Utilities for working with the local dataset cache.
This file is adapted from the AllenNLP library at https://github.com/allenai/allennlp
Copyright by the AllenNLP authors.
"""

import os
import sys
import tempfile
import shutil
import logging
from pathlib import Path
from urllib.parse import urlparse

from .file_utils import _http_get

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(level=logging.INFO)
FORMAT = logging.Formatter("%(levelname)s: %(message)s")
SH = logging.StreamHandler(stream=sys.stdout)
SH.setFormatter(FORMAT)
LOGGER.addHandler(SH)
CACHE_DIR = os.path.join(str(Path.home()), '.cotk_cache')

def load_model_from_url(url, cache_dir=CACHE_DIR):
	'''Download model at the given URL and save it in CACHE_DIR.
	return: str, the local path of downloaded model
	Example:
		>>> load_model_from_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')
		>>> {CACHE_DIR}/resnet18-5c106cde.pth
	'''
	parts = urlparse(url)
	filename = os.path.basename(parts.path)

	cache_dir = os.path.join(cache_dir, 'models')
	os.makedirs(cache_dir, exist_ok=True)
	cache_path = os.path.join(cache_dir, filename)
	if os.path.exists(cache_path):
		raise ValueError("model existed. If you want to delete the existing model. \
			Use `rm %s`." % cache_path)

	with tempfile.NamedTemporaryFile() as temp_file:
		_http_get(url, temp_file)
		temp_file.flush() # flush to avoid truncation
		temp_file.seek(0) # shutil.copyfileobj() starts at the current position

		with open(cache_path, 'wb') as cache_file:
			shutil.copyfileobj(temp_file, cache_file)

	LOGGER.info('model cached at %s', cache_path)
	return cache_path
