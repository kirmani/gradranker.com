#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 Sean Kirmani <sean@kirmani.io>
#
# Distributed under terms of the MIT license.

import os

DATA_FOLDER_NAME = 'data'

SCHOOLS_FILE_NAME = 'schools.json'

FIELDS_FILE_NAME = 'fields.json'
FIELD_QUERY_KEY = 'query'
FIELD_DISPLAY_NAME_KEY = 'display_name'

APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
APP_STATIC = os.path.join(APP_ROOT, 'static')
DATA_DIR_PATH = os.path.join(APP_ROOT, DATA_FOLDER_NAME)
SCHOOLS_FILE_PATH = os.path.join(DATA_DIR_PATH, SCHOOLS_FILE_NAME)
FIELDS_FILE_PATH = os.path.join(DATA_DIR_PATH, FIELDS_FILE_NAME)
