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
# MAGIC We download text content for online CSR disclosures using the `PyPDF2` library. Please note that you will need to support outbound HTTP access from your databricks workspace. Although having a central place where to source data from (such as [responsibilityreports.com](https://www.responsibilityreports.com)) minimizes the amount of firewall rules to enable, this approach comes at a price: it prevents user from distributing that scraping logic across multiple executors. In our approach, we download data sequentially, checkpointing to delta every x PDF documents. Just like many web scraping processes, please proceed with extra caution and refer to responsibilityreports.com [T&Cs](https://www.responsibilityreports.com/Disclaimer) before doing so.

# COMMAND ----------

from pyspark.sql import functions as F
import pandas as pd

def save_csr_content(csr_data, i, n):
  # create a dataframe for each batch of downloaded reports
  df = pd.DataFrame(csr_data, columns=['organization', 'sector', 'ticker', 'url', 'content'])
  # create a new view
  sdf = spark.createDataFrame(df).filter(F.length('content') > 0)
  # store batch of records to delta table
  sdf.write.format('delta').mode('append').saveAsTable(csr_table_bronze)
  print("Downloaded {}/{}".format(i + 1, n))
  # clean our checkpoint
  csr_data.clear()

# COMMAND ----------

# MAGIC %md
# MAGIC In order to guarantee consistency between different releases and enable unit tests, we moved the scraper logic to an arbitrary python file loaded as `scraper_utils` module here.

# COMMAND ----------

from utils.scraper_utils import *

# COMMAND ----------

for sector in sectors.keys():
  
  print('')
  print('*'*50)
  print('Accessing reports for [{}]'.format(sectors[sector]))
  print('*'*50)
  print('')
  
  csr_data = []
  organizations = get_organizations(sector)
  for i, organization in enumerate(organizations):
    [name, ticker, url] = get_organization_details(organization)
    content = download_csr(url)
    csr_data.append([name, sectors[sector], ticker, url, content])
    if i > 0 and i % config['csr']['batch_size'] == 0:
      save_csr_content(csr_data, i, len(organizations))
  if len(csr_data) > 0:
    save_csr_content(csr_data,  i, len(organizations))

# COMMAND ----------

display(spark.read.table(csr_table_bronze))

# COMMAND ----------

# MAGIC %md
# MAGIC Since we've stored many chunks of data, we may want to optimize our table and delete previous versions. This is done through the `OPTIMIZE` and `VACUUM` commands respectively, recently released to open source through our Delta 2.0 [announcement](https://databricks.com/blog/2022/06/30/open-sourcing-all-of-delta-lake.html).

# COMMAND ----------

_ = sql("OPTIMIZE {}".format(csr_table_bronze))
_ = sql("VACUUM {}".format(csr_table_bronze))
display(spark.read.table(csr_table_bronze))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract sentences
# MAGIC PDFs are highly unstructured by nature with text that is often scattered across multiple lines, pages, columns. From a simple set of regular expressions to a more complex NLP model (we use a [nltk](https://www.nltk.org/) trained pipeline), we show how to extract clean sentences from raw text documents in our utility functions. 

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
from typing import Iterator
import pandas as pd
from utils.nlp_utils import *

@pandas_udf('array<string>')
def get_sentences_from_csr(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    load_nltk(nltk_path)
    for xs in batch_iter:
        yield xs.apply(extract_statements)

# COMMAND ----------

csr_raw_df = spark.read.table(csr_table_bronze)

# COMMAND ----------

from pyspark.sql.functions import col, length, explode

display(
  csr_raw_df
    .withColumn('content', get_sentences_from_csr(col('content')))
    .withColumn('statement', explode(col('content')))
    .filter(length('statement') > 255)
    .select('organization', 'ticker', 'sector', 'url', 'statement')
    .write
    .format('delta')
    .mode('overwrite')
    .saveAsTable(csr_table_silver)
)

# COMMAND ----------

_ = sql("OPTIMIZE {} ZORDER BY organization".format(csr_table_silver))
display(spark.read.table(csr_table_silver))

# COMMAND ----------


