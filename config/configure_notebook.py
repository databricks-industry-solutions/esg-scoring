# Databricks notebook source
# MAGIC %pip install -r requirements.txt
# MAGIC %pip uninstall -y pydantic # Issue with spacy library
# MAGIC %pip install pydantic
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

############ UPDATE CONFIGURATION AS REQUIRED
 
catalog_name = 'industry_solutions'
schema_name = 'esg_scoring'
cache_volume = 'cache' 
data_volume = 'data' 
gkg_marketplace_table = 'esg_gdelt.cx13214.`gkg_v1_daily__aqlonmvwi0cqjpbr9ajyg-vvaq`'
model_name = 'esg_lda'
num_executors = 6

# COMMAND ----------

csr_table_content = f'{catalog_name}.{schema_name}.csr_content'
csr_table_statement = f'{catalog_name}.{schema_name}.csr_statement'
csr_table_topics = f'{catalog_name}.{schema_name}.csr_topic'
csr_table_scores = f'{catalog_name}.{schema_name}.csr_scores'
csr_table_gold = f'{catalog_name}.{schema_name}.csr_gold'

gdelt_bronze_table = f'{catalog_name}.{schema_name}.gdelt'
portfolio_table = f'{catalog_name}.{schema_name}.portfolio'

spacy_path = f'/Volumes/{catalog_name}/{schema_name}/{cache_volume}/spacy'
nltk_path = f'/Volumes/{catalog_name}/{schema_name}/{cache_volume}/nltk'
data_path = f'/Volumes/{catalog_name}/{schema_name}/{data_volume}'

model_registered_name = f'{catalog_name}.{schema_name}.{model_name}'

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

_ = sql(f'CREATE CATALOG IF NOT EXISTS {catalog_name}')
_ = sql(f'CREATE DATABASE IF NOT EXISTS {catalog_name}.{schema_name}')
_ = sql(f'CREATE VOLUME IF NOT EXISTS {catalog_name}.{schema_name}.{cache_volume}')
_ = sql(f'CREATE VOLUME IF NOT EXISTS {catalog_name}.{schema_name}.{data_volume}')

# COMMAND ----------

import os
import spacy 
import nltk

if not os.path.exists(spacy_path):
  print("Downloading SPACY model to {}".format(spacy_path))
  os.mkdir(spacy_path)
  spacy.cli.download("en_core_web_sm")
  nlp = spacy.load('en_core_web_sm')
  nlp.to_disk(spacy_path)

if not os.path.exists(nltk_path):
  print("Downloading NLTK model to {}".format(nltk_path))
  os.mkdir(nltk_path)
  nltk.download('wordnet', download_dir="{}/wordnet".format(nltk_path))
  nltk.download('punkt', download_dir="{}/punkt".format(nltk_path))
  nltk.download('omw-1.4', download_dir="{}/omw".format(nltk_path))
