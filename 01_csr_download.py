# Databricks notebook source
# MAGIC %md
# MAGIC # Access CSR reports
# MAGIC In this notebook, we search for available corporate sustainability reports from publicly traded organizations. Instead of going through each company website, we will access information from [responsibilityreports.com](https://www.responsibilityreports.com) and download each CSR report across multiple sectors. Please be aware that the quality of analytics derived in that solution strongly depends on the amount of PDF documents to learn from, so having a clear data access strategy is key to ESG success. 

# COMMAND ----------

# MAGIC %run ./config/configure_notebook

# COMMAND ----------

csr_table_bronze = config['database']['tables']['csr']['bronze']
csr_table_silver = config['database']['tables']['csr']['silver']

# COMMAND ----------

sectors = {
  3 : 'Consumer goods',
  4 : 'Financial Services',
  5 : 'Healthcare',
  7 : 'All Services',
  8 : 'Technology companies',
  24: 'Energy'
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download CSRs
# MAGIC We download text content for online CSR disclosures using the `PyPDF2` library. Please note that you will need to support outbound HTTP access from your databricks workspace. Although having a central place where to source data from (such as [responsibilityreports.com](https://www.responsibilityreports.com)) minimizes the amount of firewall rules to enable, this approach comes at a price: it prevents user from distributing that scraping logic across multiple executors. In our approach, we download data sequentially, checkpointing to delta every 20 PDF documents. Just like many web scraping processes, please proceed with extra caution and refer to responsibilityreports.com [T&Cs](https://www.responsibilityreports.com/Disclaimer) before doing so.

# COMMAND ----------

from bs4 import BeautifulSoup
import requests
import urllib.request
import pandas as pd

def get_organizations(sector):
  index_url = "https://www.responsibilityreports.com/Companies?sect={}".format(sector)
  response = requests.get(index_url)
  soup = BeautifulSoup(response.text,"html.parser")
  csr_entries = [link.get('href') for link in soup.findAll('a')]
  organizations = [ele.split("/")[-1] for ele in csr_entries if ele.startswith('/Company/')]
  print('Found {} organization(s)'.format(len(organizations)))
  return organizations

# COMMAND ----------

def get_organization_details(organization):
  # use beautiful soup to parse company page on responsibilityreports.com
  company_url = "https://www.responsibilityreports.com/Company/" + organization
  response = requests.get(company_url)
  soup = BeautifulSoup(response.text)
  try:
    # page contains useful information such as company legal name and ticker
    name = soup.find('h1').text
    ticker = soup.find('span', {"class": "ticker_name"}).text
    csr_url = ""
    # also contains the link to their most recent disclosures
    for link in soup.findAll('a'):
      data = link.get('href')
      if data.split('.')[-1]=='pdf':
        csr_url = 'https://www.responsibilityreports.com'+data
        break
    return [name, ticker, csr_url]
  except:
    # a lot of things could go wrong here, simply ignore that record
    return ["", "", ""]

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
    return ""

# COMMAND ----------

from pyspark.sql import functions as F

def save_csr_content(csr_data, i, n):
  # create a dataframe for each batch of downloaded reports
  df = pd.DataFrame(csr_data, columns=['organization', 'sector', 'ticker', 'url', 'content'])
  # create a new view
  sdf = spark.createDataFrame(df).filter(F.length('content') > 0)
  # store bactch of records to delta table
  sdf.write.format('delta').mode('append').saveAsTable(csr_table_bronze)
  print("Downloaded {}/{}".format(i + 1, n))
  # clean our checkpoint
  csr_data.clear()

# COMMAND ----------

for sector in sectors.keys():
  
  print('')
  print('*'*60)
  print('Accessing reports for [{}]'.format(sectors[sector]))
  print('*'*60)
  print('')
  
  csr_data = []
  organizations = get_organizations(sector)
  for i, organization in enumerate(organizations):
    [name, ticker, url] = get_organization_details(organization)
    content = get_csr_content(url)
    csr_data.append([name, sectors[sector], ticker, url, content])
    if i > 0 and i % config['csr']['batch_size'] == 0:
      save_csr_content(csr_data, i, len(organizations))
  if len(csr_data) > 0:
    save_csr_content(csr_data,  i, len(organizations))

# COMMAND ----------

# MAGIC %md
# MAGIC Since we've stored many chunks of data, we may want to optimize our table and delete previous versions. This is done through the `OPTIMIZE` and `VACUUM` commands respectively.

# COMMAND ----------

_ = sql("OPTIMIZE {}".format(csr_table_bronze))
_ = sql("VACUUM {}".format(csr_table_bronze))
display(spark.read.table(csr_table_bronze))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract sentences
# MAGIC PDFs are highly unstructured by nature with text that is often scattered across multiple lines, pages, columns. From a simple set of regular expressions to a more complex NLP model (we use a [nltk](https://www.nltk.org/) trained pipeline), we show how to extract clean sentences from raw text documents. 

# COMMAND ----------

from utils.nlp_utils import *

# COMMAND ----------

csr_raw_df = spark.read.table(csr_table_bronze)

# COMMAND ----------

import nltk
import re

def clean_line(line):
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
  return [str(part).strip().lower() for part in nltk.sent_tokenize(line)]

# COMMAND ----------

import string

def extract_statements(text):
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
    sentences.extend(clean_line(line))
  return sentences

# COMMAND ----------

from typing import Iterator
from pyspark.sql.functions import pandas_udf
import pandas as pd

@pandas_udf('array<string>')
def clean_text(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    load_nltk(nltk_path)
    for xs in batch_iter:
      yield xs.apply(extract_statements)

# COMMAND ----------

from pyspark.sql.functions import col, explode

display(
  csr_raw_df
    .withColumn('content', clean_text(col('content')))
    .withColumn('statement', explode(col('content')))
    .select('organization', 'ticker', 'sector', 'url', 'statement')
    .write
    .format('delta')
    .mode('overwrite')
    .saveAsTable(csr_table_silver)
)

# COMMAND ----------

_ = sql("OPTIMIZE {} ZORDER BY organization".format(csr_table_silver))
display(spark.read.table(csr_table_silver))
