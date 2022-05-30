# Databricks notebook source
# MAGIC %pip install -r requirements.txt

# COMMAND ----------

import yaml
with open('config/application.yaml', 'r') as f:
  config = yaml.safe_load(f)

# COMMAND ----------

dbutils.fs.mkdirs(config['database']['path'])
_ = sql("CREATE DATABASE IF NOT EXISTS {} LOCATION '{}'".format(
  config['database']['name'], 
  config['database']['path']
))

# COMMAND ----------

# use our newly created database by default
# each table will be created as a MANAGED table under this directory
_ = sql("USE {}".format(config['database']['name']))

# COMMAND ----------

import os
import spacy 
import nltk

# we expect models to be available on a mounted directory accessible by all executors
# one can call those methods to download models if needed
spacy_path = config['model']['spacy']['path']
if not os.path.exists(spacy_path):
  os.mkdir(spacy_path)
  spacy.cli.download("en_core_web_sm")
  nlp = spacy.load('en_core_web_sm')
  nlp.to_disk(spacy_path)

  
nltk_path = config['model']['nltk']['path']
if not os.path.exists(nltk_path):
  os.mkdir(nltk_path)
  
if not os.path.exists("{}/wordnet".format(nltk_path)):
  nltk.download('wordnet', download_dir="{}/wordnet".format(nltk_path))
  
if not os.path.exists("{}/punkt".format(nltk_path)):
  nltk.download('punkt', download_dir="{}/punkt".format(nltk_path))

# COMMAND ----------

with open('config/portfolio.txt', 'r') as f:
  portfolio = f.read().split('\n')

# COMMAND ----------

# np.random.RandomState was deprecated, so Hyperopt now uses np.random.Generator
import hyperopt
import numpy as np

if hyperopt.__version__.split('+')[0] > '0.2.5':
  rstate=np.random.default_rng(123)
else:
  rstate=np.random.RandomState(123)

# COMMAND ----------

def teardown():
  _ = sql("DROP DATABASE IF EXISTS {} CASCADE".format(config['database']['name']))
  dbutils.fs.rm(config['database']['path'], True)
  dbutils.fs.rm(config['model']['spacy']['path'], True)
  dbutils.fs.rm(config['model']['nltk']['path'], True)
