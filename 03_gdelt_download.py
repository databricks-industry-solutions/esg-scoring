# Databricks notebook source
# MAGIC %md
# MAGIC # Access news articles
# MAGIC As covered in the previous section, we were able to compare businesses side by side across different ESG initiatives. However, we do not want to solely base our assumptions on companies’ official disclosures but rather on how companies' reputation is perceived in the media, across all 3 environmental, social and governance variables. In this notebook, we analyze news articles to understand companies' ESG initiatives beyond their immediate disclosures.

# COMMAND ----------

# MAGIC %run ./config/configure_notebook

# COMMAND ----------

gdelt_bquery_table = config['bigquery']['gdelt']
gdelt_bronze_table = config['database']['tables']['gdelt']['bronze']

# COMMAND ----------

# MAGIC %md
# MAGIC ## News dataset
# MAGIC [GDELT](https://www.gdeltproject.org/) Project monitors the world's broadcast, print, and web news from nearly every corner of every country in over 100 languages and identifies the people, locations, organizations, themes, sources, emotions, counts, quotes, images and events driving our global society every second of every day, creating a free open platform for computing on the entire world. Supported by Google Jigsaw, GDELT datasets are available on Google BigQuery or flat files with new increment available every 15mn

# COMMAND ----------

from pyspark.sql import functions as F

gdelt_raw = (
  spark
    .read
    .format("bigquery")
    .option("table", gdelt_bquery_table)
    .load()
    .filter(F.col('DATE') >  config['gdelt']['mindate'])
    .filter(F.col('DATE') <= config['gdelt']['maxdate'])
)

# COMMAND ----------

display(gdelt_raw)

# COMMAND ----------

# GDELT comes with a prebuilt taxonomy to detect thousands of themes
# Although noisy (keywords based), pre-filtering for themes would drastically reduce our dataset
# We search for United Nation Guiding Principles (S), Economy related news (G) and environmental (E) as first proxy 
gdelt_urls = (
  gdelt_raw
    .withColumnRenamed('DocumentIdentifier', 'url')
    .withColumn('theme', F.explode(F.split('Themes', ';')))
    .filter(F.split(F.col('theme'), '_')[0].isin(['ENV', 'ECON', 'UNGP']))
    .select('url')
    .distinct()
)

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
from utils.nlp_utils import *
from typing import Iterator
import pandas as pd

@pandas_udf('string')
def lemmatize_text_udf(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
  load_nltk(nltk_path)
  for xs in batch_iter:
    yield xs.apply(lemmatize_text)

# COMMAND ----------

gdelt_esg = (
  gdelt_raw
    .withColumn('title', F.regexp_extract(F.col('Extras'), '<PAGE_TITLE>(.*)</PAGE_TITLE>', 1))
    .withColumn('publishDate', F.to_timestamp(F.expr('CAST(DATE AS STRING)'), 'yyyyMMddHHmmss'))
    .select(
      F.col('publishDate'),
      F.split(F.col('Organizations'), ';').alias('organizations'),
      F.col('DocumentIdentifier').alias('url'),
      F.col('SourceCommonName').alias('source'),
      F.col('title'),
      F.split(F.col('V2Tone'), ',')[0].alias('tone')
    )
    .filter(F.size(F.col('organizations')) > 0)                 # at least one organization must be mentioned
    .join(gdelt_urls, ['url'])                                  # only retrieve news with esg theme candidates
    .filter(F.length(lemmatize_text_udf(F.col('title'))) > 50)  # title should have been extracted
)

display(gdelt_esg)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Organization contribution
# MAGIC Some organizations may be mentioned in the news via their common names (e.g. IBM) rather than their legal entities (e.g. International Business Machines Corporation), resulting in an overall poor match when joining our datasets "as-is". Furthermore, many organizations are related to others, such as portfolio companies belonging to a same holding. Analysing ESG initiatives must take the global dynamics of a business into account. For that purpose, we will use NLP techniques to extract mentions of any organizations in a given CSR report and link those "named entities" to better capture ESG related news we know are relevant for any given organization.

# COMMAND ----------

@pandas_udf('array<string>')
def extract_organizations_udf(content_series_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
  nlp = load_spacy(spacy_path)
  for content_series in content_series_iter:
    yield content_series.map(lambda text: extract_organizations(text, nlp))

# COMMAND ----------

from pyspark.sql import functions as F

org_relations = (
  spark
    .read
    .table(config['database']['tables']['csr']['bronze'])
    .filter(F.col('ticker').isin(portfolio))
    .select('ticker', 'content')
    .select(
      F.col('ticker'),
      F.explode(extract_organizations_udf(F.col('content'))).alias('organization')
    )
    .filter(F.length(F.col('organization')) >= 3)
)

# COMMAND ----------

# how many times did organization A mentioned organization B in their reports
org_contribution_tf = (
  org_relations
    .groupBy('ticker', 'organization')
    .count()
    .cache()
)

# how important is organization B is relative to other
org_contribution = (
  org_contribution_tf
    .groupBy('ticker')
    .agg(F.sum('count').alias('mentions'))
    .join(org_contribution_tf, ['ticker'])
    .select(
      F.col('ticker'),
      F.col('organization'),
      (F.col('count') / F.col('mentions')).alias('org_contribution')
    )
)

# COMMAND ----------

display(org_contribution.orderBy(F.desc('org_contribution')))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Media contribution
# MAGIC As most news articles would be mentioning more than one organization, we need a similar weighting strategy. Furthermore, many businesses (especially in the media industry) might be overindexed, resulting in irrelevant insights and probably low ESG scores. We use a simple term frequency analysis (TF-IDF) to understand the significance of each organization to a given article and attribute its impact proortionally. 

# COMMAND ----------

@udf('string')
def clean_org_name_udf(text):
  return clean_org_name(text)

# COMMAND ----------

# how many times was organization A mentioned in that article
gkg_org_tf = (
  gdelt_raw
    .select(
      F.split(F.col('Organizations'), ';').alias('organizations'), 
      F.col('DocumentIdentifier').alias('url')
    )
    .withColumn('organization', F.explode(F.col('organizations')))
    .withColumn('organization', clean_org_name_udf(F.col('organization')))
    .groupBy('organization', 'url')
    .count()
    .withColumnRenamed('count', 'tf')
    .cache()
)

# COMMAND ----------

# how many articles mentioned organization A
gkg_org_df = gkg_org_tf.groupBy('organization').count().withColumnRenamed('count', 'df')

# COMMAND ----------

# significance of organization A for a given article
gkg_n = gkg_org_tf.select('url').distinct().count()
gkg_org_tfidf = (
  gkg_org_tf.join(gkg_org_df, ['organization'])
    .select(
      F.col('organization'),
      F.col('url'),
      (F.col('df') * F.log(F.lit(gkg_n) / F.col('df'))).alias('tfidf')
    )
)

# COMMAND ----------

# how much did article cover organization A relative to others
gkg_contribution = (
  gkg_org_tfidf
    .groupBy('url')
    .agg(F.sum('tfidf').alias('sum_tfidf'))
    .join(gkg_org_tfidf, ['url'])
    .select(
      F.col('url'),
      F.col('organization'),
      (F.col('tfidf') / F.col('sum_tfidf')).alias('gkg_contribution')
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Media coverage
# MAGIC Finally, we can join our weighting factors together and access each news article tagged with organizations' tickers. Although we may have wrongly assigned some articles to some organizations (given the GDELT signal over noise ratio and NLP limited accuracy), the side effects of attributing a news to the wrong organization will be limited given our contribution factors on both organization and media coverage. We summarize our weighting strategy below.
# MAGIC 
# MAGIC <img src='https://d1r5llqwmkrl74.cloudfront.net/notebooks/fsi/esg_scoring/gcp/images/news_contribution.png' width=300>

# COMMAND ----------

esg_coverage = (
  gkg_contribution
    .join(org_contribution, ['organization'])
    .select(
      F.col('ticker'),
      F.col('url'),
      F.col('org_contribution'),
      F.col('gkg_contribution')
    )
)

# COMMAND ----------

_ = (
  gdelt_esg.join(esg_coverage, ['url']).select(
    'publishDate',
    'ticker',
    'source',
    'url',
    'title',
    'tone',
    'org_contribution',
    'gkg_contribution'
  )
  .write
  .format('delta')
  .mode('overwrite')
  .saveAsTable(gdelt_bronze_table)
)

# COMMAND ----------

_ = sql("OPTIMIZE {} ZORDER BY ticker".format(gdelt_bronze_table))

# COMMAND ----------

n = spark.read.table(gdelt_bronze_table).count()
displayHTML('<p>We now have access to {} news articles covering our portfolio</p>'.format(n))
