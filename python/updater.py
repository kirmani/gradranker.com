#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 Sean Kirmani <sean@kirmani.io>
#
# Distributed under terms of the MIT license.
"""TODO(Sean Kirmani): DO NOT SUBMIT without one-line documentation for test

TODO(Sean Kirmani): DO NOT SUBMIT without a detailed description of test.
"""
import argparse
import json
import os
import re
import sys
import time
import traceback

import constants

APP_ROOT = os.path.dirname(os.path.abspath(__file__))   # refers to application_top

def main():
  global args
  if args.field:
    action = args.field[0]
    if action == 'add':
      key = args.field[1]
      query = args.field[2]
      display_name = args.field[3]
      fields = _GetFields()
      _AddField(fields, key, query, display_name)
      _UpdateFieldsFile(fields)
    elif action == 'remove':
      key = args.field[1]
      fields = _GetFields()
      _RemoveField(fields, key)
      _UpdateFieldsFile(fields)
    else:
      print("%s is not a valid action for field" % action)


def _GetFields():
  with open(constants.FIELDS_FILE_PATH, 'rb') as f:
    return json.load(f)
  return None

def _UpdateFieldsFile(fields):
  with open(constants.FIELDS_FILE_PATH,'w+') as f:
    f.seek(0)
    json.dump(fields, f, indent=4)

def _HasField(fields, key):
  has_field = key in fields
  if args.verbose:
    if has_field:
      print("%s in fields." % key)
    else:
      print("%s not in fields." % key)
  return has_field

def _AddField(fields, key, query, display_name):
  if args.verbose: print("Attempting to add %s to fields." % key)
  if _HasField(fields, key):
    if args.verbose: print("Failed to add %s to fields." % key)
  else:
    fields[key] = {
        constants.FIELD_QUERY_KEY : query,
        constants.FIELD_DISPLAY_NAME_KEY : display_name
        }
    print("Added %s to fields. %s" % (key, fields[key]))

def _RemoveField(fields, key):
  if args.verbose: print("Attempting to add %s to fields." % key)
  if _HasField(fields, key):
    fields.pop(key)
    if args.verbose: print("Removed %s from fields." % key)
  else:
    print("Failed to remove %s from fields." % key)

if __name__ == '__main__':
  try:
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-v','--verbose', action='store_true', default=False, \
        help='verbose output')
    parser.add_argument('-d','--debug', action='store_true', default=False, \
        help='debug output')
    parser.add_argument('-f','--field', nargs='*', dest='field',
        help='adds FIELD to the list of fields', metavar='FIELD')
    args = parser.parse_args()
    # if len(args) < 1:
    #   parser.error('missing argument')
    if args.verbose: print(time.asctime())
    main()
    if args.verbose: print(time.asctime())
    if args.verbose: print('TOTAL TIME IN MINUTES:')
    if args.verbose: print(time.time() - start_time) / 60.0
    sys.exit(0)
  except KeyboardInterrupt, e: # Ctrl-C
    raise e
  except SystemExit, e: # sys.exit()
    raise e
  except Exception, e:
    print('ERROR, UNEXPECTED EXCEPTION')
    print(str(e))
    traceback.print_exc()
    os._exit(1)
