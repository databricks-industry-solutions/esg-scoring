# Databricks notebook source
import nltk
import os

def load_nltk():
  nltk.data.path.append("/dbfs{}/wordnet".format(data_path))
  nltk.data.path.append("/dbfs{}/punkt".format(data_path))

dbutils.fs.mkdirs("{}/wordnet".format(data_path))
dbutils.fs.mkdirs("{}/punkt".format(data_path))
# as we're distributing that scraping process, existing NLP pipelines used to 
# extract sentences must be made accessible across executors (hence on /dbfs) 
nltk.download('wordnet', download_dir="/dbfs{}/wordnet".format(data_path))
nltk.download('punkt', download_dir="/dbfs{}/punkt".format(data_path))

# COMMAND ----------

import requests
from PyPDF2 import PdfFileReader
from io import BytesIO

def get_csr_content(url):
  try:
    # extract plain text from online PDF document
    response = requests.get(url)
    open_pdf_file = BytesIO(response.content)
    pdf = PdfFileReader(open_pdf_file, strict=False)
    # simply concatenate all pages as we'll clean it up later
    text = [pdf.getPage(i).extractText() for i in range(0, pdf.getNumPages())]
    return "\n".join(text)
  except:
    # a lot of things could go wrong here, simply ignore that record
    # we found that < 10% of links could not be read because of different PDF encodings
    # we may decide that entire pipeline to fail for a missing CSR report (handy for integration test)
    if config['csr_fail_if_404']:
      raise Exception(f'Report {url} could not be downloaded')
    return ""

# COMMAND ----------

import spacy

spacy.cli.download("en_core_web_sm")
nlp = spacy.load('en_core_web_sm')
nlp.to_disk("/dbfs{}/spacy".format(data_path))

def load_spacy():
    import spacy
    return spacy.load("/dbfs{}/spacy".format(data_path))

# COMMAND ----------

import re
from gensim.parsing.preprocessing import STOPWORDS

def clean_org_name(text):
    text = text.lower()
    name = []
    stop_words = {'group', 'inc', 'ltd', 'ag', 'plc', 'limited', 'sa', 'holdings'}
    stop_words = set(STOPWORDS.union(stop_words))
    for t in re.split('\\W', text):
        if len(t) > 0 and t not in stop_words:
            name.append(t)
    if len(name) > 0:
        return ' '.join(name).strip()
    else:
        return ''

# COMMAND ----------

def extract_organizations(text, nlp):
    doc = nlp(text)
    orgs = [X.text for X in doc.ents if X.label_ == 'ORG']
    return [clean_org_name(org) for org in orgs]

# COMMAND ----------

import string
from pyspark.sql import functions as F
import re

def _extract_statements(text):

  # remove non ASCII characters
  printable = set(string.printable)
  text = ''.join(filter(lambda x: x in printable, text))
  
  lines = []
  prev = ""
  for line in text.split('\n'):
    # aggregate consecutive lines where text may be broken down
    # only if next line starts with a space or previous does not end with a dot.
    if(line.startswith(' ') or not prev.endswith('.')):
        prev = prev + ' ' + line
    else:
        # new paragraph
        lines.append(prev)
        prev = line
        
  # don't forget left-over paragraph
  lines.append(prev)

  # clean paragraphs from extra space, unwanted characters, urls, etc.
  # best effort clean up, consider a more versatile cleaner
  sentences = []
  
  for line in lines:
      # removing header number
      line = re.sub(r'^\s?\d+(.*)$', r'\1', line)
      # removing trailing spaces
      line = line.strip()
      # words may be split between lines, ensure we link them back together
      line = re.sub(r'\s?-\s?', '-', line)
      # remove space prior to punctuation
      line = re.sub(r'\s?([,:;\.])', r'\1', line)
      # ESG contains a lot of figures that are not relevant to grammatical structure
      line = re.sub(r'\d{5,}', r' ', line)
      # remove mentions of URLs
      line = re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', r' ', line)
      # remove multiple spaces
      line = re.sub(r'\s+', ' ', line)
      # remove multiple dot
      line = re.sub(r'\.+', '.', line)
      
      # split paragraphs into well defined sentences using nltk
      for part in nltk.sent_tokenize(line):
        sentences.append(str(part).strip())

  return sentences

@F.udf('array<string>')
def extract_statements(content):
  load_nltk()
  return _extract_statements(content)

# COMMAND ----------

import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from gensim.utils import simple_preprocess

def _lemmatize_text(text):
  results = []
  lemmatizer = WordNetLemmatizer()
  stemmer = PorterStemmer()
  for token in simple_preprocess(text):
    stem = stemmer.stem(lemmatizer.lemmatize(token))
    if (len(stem) > 3):
      results.append(stem)
  return ' '.join(results)

@F.udf('string')
def lemmatize_text(content):
  load_nltk()
  return _lemmatize_text(content)

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import *
from pyspark.sql import functions as F

schema = ArrayType(StructType([
    StructField("id", IntegerType(), False),
    StructField("probability", FloatType(), False)
]))

@udf(schema)
def with_topic(ps):
  return [[i, p] for i, p in enumerate(ps)]
