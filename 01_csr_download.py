# Databricks notebook source
# MAGIC %md
# MAGIC # Access CSR reports
# MAGIC In this notebook, we search for available corporate sustainability reports from publicly traded organizations. Instead of going through each company website, we will access information from [responsibilityreports.com](https://www.responsibilityreports.com) and download each CSR report across multiple sectors. Please be aware that the quality of analytics derived in that solution strongly depends on the amount of PDF documents to learn from, so having a clear data access strategy is key to ESG success. 

# COMMAND ----------

# MAGIC %run ./config/configure_notebook

# COMMAND ----------

import pandas as pd
portfolio = pd.read_json('config/portfolio.json')
display(portfolio)

# COMMAND ----------

_ = spark.createDataFrame(portfolio).write.mode('overwrite').saveAsTable(portfolio_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download CSRs
# MAGIC We download text content for online CSR disclosures. Please note that you will need to support outbound HTTP access from your databricks workspace. Although having a central place where to source data from (such as [responsibilityreports.com](https://www.responsibilityreports.com)) minimizes the amount of firewall rules to enable, this approach comes at a price: it prevents user from distributing that scraping logic across multiple executors. In our approach, we download data sequentially. Just like many web scraping processes, please proceed with extra caution and refer to responsibilityreports.com [T&Cs](https://www.responsibilityreports.com/Disclaimer) before doing so.

# COMMAND ----------

import os
import shutil
import requests
from pathlib import Path

csr_urls = portfolio['url']
for csr_url in csr_urls:
  basename = os.path.basename(csr_url)
  output_file = os.path.join(data_path, basename)
  response = requests.get(csr_url)
  filename = Path(output_file)
  filename.write_bytes(response.content)

# COMMAND ----------

display(dbutils.fs.ls(data_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tika text extraction
# MAGIC Using [TikaInputFormat](https://github.com/databrickslabs/tika-ocr) library and tesseract binaries installed on each executor as an init script (optional), we can read any unstructured text as-is, extracting content type, text and metadata. Although this demo only focuses on PDF documents, Tika supports literally any single MIME type, from email, pictures, xls, html, powerpoints, scanned images, etc. Given our utility library installed on your cluster as an external [maven dependency](https://mvnrepository.com/artifact/com.databricks.labs/tika-ocr) and tesseract installed thanks to an init script (see `init.sh`), we abstracted most of its complexity away through a simple operation, `spark.read.format('tika')`

# COMMAND ----------

csr_content = spark.read.format('tika').load(data_path).cache()
display(csr_content)

# COMMAND ----------

_ = csr_content.write.mode('overwrite').saveAsTable(csr_table_content)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract sentences
# MAGIC PDFs are highly unstructured by nature with text that is often scattered across multiple lines, pages, columns. From a simple set of regular expressions to a more complex NLP model (we use a [nltk](https://www.nltk.org/) trained pipeline), we show how to extract clean sentences from raw text documents in our utility functions. 

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, udf
from pyspark.sql.functions import col, length, explode
from typing import Iterator
import pandas as pd
from utils.nlp_utils import *

@udf('string')
def get_file_name(url):
    return os.path.basename(url)

@pandas_udf('array<string>')
def extract_sentences(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    load_nltk(nltk_path)
    for xs in batch_iter:
        yield xs.apply(extract_statements)

# COMMAND ----------

portfolio = spark.read.table(portfolio_table).select(
  col('ticker'),
  get_file_name('url').alias('file')
)

# COMMAND ----------

_ = (
  csr_content
    .withColumn('content', extract_sentences(col('contentText')))
    .withColumn('statement', explode(col('content')))
    .filter(length('statement') > 255)
    .withColumn('file', get_file_name('path'))
    .join(portfolio, ['file'])
    .select('ticker', 'file', 'statement')
    .write
    .format('delta')
    .mode('overwrite')
    .saveAsTable(csr_table_statement)
)

# COMMAND ----------

display(spark.read.table(csr_table_statement))

# COMMAND ----------


