#!/usr/bin/python

import pandas as pd
import gzip
import json

def loader(path):
  """Read dataset. return iterator that dump one line string a time"""
  if path.endswith('.gz'):
    g = gzip.open(path, 'rb')
  elif path.endswith('.json'):
    g = open(path, 'r')
  else:
    raise RuntimeError("File not supported")
  for l in g:
    yield l

def parse(path):
  """Read dataset. return iterator that dump one record a time"""
  if path.endswith('.gz'):
    g = gzip.open(path, 'rb')
  elif path.endswith('.json'):
    g = open(path, 'r')
  else:
    raise RuntimeError("File not supported")
  for l in g:
    yield json.loads(l)

    
def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

if __name__ == '__main__':
    df = getDF('../../data/business_example.json')
    print(df)