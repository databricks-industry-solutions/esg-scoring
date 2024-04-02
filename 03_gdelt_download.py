# Databricks notebook source
# MAGIC %md
# MAGIC # Access news articles
# MAGIC As covered in the previous section, we were able to compare businesses side by side across different ESG initiatives. However, we do not want to solely base our assumptions on companies’ official disclosures but rather on how companies' reputation is perceived in the media, across all 3 environmental, social and governance variables. In this notebook, we analyze news articles to understand companies' ESG initiatives beyond their immediate disclosures.

# COMMAND ----------

# MAGIC %run ./config/configure_notebook

# COMMAND ----------

# MAGIC %md
# MAGIC ## News dataset
# MAGIC Supported by Google Jigsaw, the [GDELT](https://www.gdeltproject.org/) Project monitors the world's broadcast, print, and web news from nearly every corner of every country in over 100 languages and identifies the people, locations, organizations, themes, sources, emotions, counts, quotes, images and events driving our global society every second of every day, creating a free open platform for computing on the entire world. Gdelt files are available on databricks marketplace free of charge (Published by Crux). 

# COMMAND ----------

gkg = spark.read.table(gkg_marketplace_table)
display(gkg.limit(100))

# COMMAND ----------

# MAGIC %md
# MAGIC The structure of GDELT GKGv2 files is complex (see [data model](http://data.gdeltproject.org/documentation/GDELT-Global_Knowledge_Graph_Codebook-V2.1.pdf)). Because of its complex taxonomy, some fields may be separated by comma, spaces and nested entities separated by hash or semi columns. 

# COMMAND ----------

from utils.nlp_utils import *
from pyspark.sql.functions import udf

@udf('string')
def clean_org_name_udf(text):
  return clean_org_name(text)

# COMMAND ----------

# GDELT comes with a prebuilt taxonomy to detect thousands of themes
# Although noisy (keywords based), filtering for themes drastically reduces our dataset
# We search for United Nation Guiding Principles as proxy for ESG related articles 
from pyspark.sql import functions as F

gdelt_articles = (
  gkg
    .withColumn('themes', F.split('THEMES', ';'))
    .withColumn('theme', F.explode('themes'))
    .filter(F.split(F.col('theme'), '_')[0] == 'UNGP')
    .withColumn('organization', F.split('ORGANIZATIONS', ';'))
    .withColumn('organization', F.explode('organization'))
    .withColumn('organization', clean_org_name_udf('organization'))
    .withColumn('source', F.split('SOURCEURLS', '<UDIV>'))
    .withColumn('source', F.explode('source'))
    .withColumn('tone', F.split('tone', ',')[0].cast('DOUBLE'))
    .select('date', 'source', 'organization', 'theme', 'tone')
)

display(gdelt_articles.limit(100))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Organization contribution
# MAGIC Some organizations may be mentioned in the news via their common names (e.g. IBM) rather than their legal entities (e.g. International Business Machines Corporation), resulting in an overall poor match when joining our datasets "as-is". Furthermore, many organizations are related to others, such as portfolio companies belonging to a same holding. Analysing ESG initiatives must take the global dynamics of a business into account. For that purpose, we will use NLP techniques to extract mentions of any organizations in a given CSR report and link those "named entities" to better capture ESG related news we know are relevant for any given organization.

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
from typing import Iterator
import pandas as pd

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
    .table(csr_table_statement)
    .select('ticker', 'statement')
    .select(
      F.col('ticker'),
      F.explode(extract_organizations_udf(F.col('statement'))).alias('organization')
    )
    .withColumn('organization', clean_org_name_udf('organization'))
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
    .cache()
)

# COMMAND ----------

display(org_contribution.limit(20))

# COMMAND ----------

organization_names = set(org_contribution.toPandas()['organization'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Media coverage
# MAGIC Ideally, we should scrape news content and apply more extensive NLP (e.g. John Snow Labs model) and genAI techniques to extract and match organizations to ticker, possibly leveraging vector store and embedding functionalities. In practice, we assume the gdelt universe to be large enough to capture enough information for organizations we could match as a simple JOIN, providing our user with a foundation towards their solutions rather than a product on its own. We invite our user to delve into the GDELT universe further. 

# COMMAND ----------

gdelt_articles_candidates = gdelt_articles.filter(F.col('organization').isin(organization_names))

# COMMAND ----------

_ = (
  gdelt_articles_candidates.join(org_contribution, ['organization'])
  .withColumnRenamed('organization', 'mention')
  .join(spark.read.table(portfolio_table), ['ticker'])
  .withColumn('tone', F.col('tone') * F.col('org_contribution'))
  .select(
    'date',
    'ticker',
    'organization',
    'mention',
    'source',
    'theme',
    'tone'
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
displayHTML('<p>We now have access to {} news article candidates covering our portfolio</p>'.format(n))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Walking the talk
# MAGIC Finally, we can combine our insights generated from our 2 notebooks to get a sense of how much each companies' initiative were possibly followed through and what was the media coverage (positive or negative). Given a data driven approach (rather than subjective scoring going through a PDF document), we can detect organisations "walking the talk". One can select a given ticker, sector or industry, alongside a specific United Nation Goal to get a more holistic view on companies strategic priorities.

# COMMAND ----------

gdelt = spark.read.table(gdelt_bronze_table)

# COMMAND ----------

display(gdelt.limit(20))

# COMMAND ----------

display(gdelt.groupBy('date').agg(F.avg('tone').alias('tone')))

# COMMAND ----------

display(gdelt.groupBy('date').count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Take Away
# MAGIC Throughout this series of notebooks, we introduced a novel approach to environmental, social and governance to objective quantify the ESG impact of public organisations using AI. By combining corporate disclosure and news analytics data, we demonstrated how machine learning could be used to bridge the gap between what companies say and what companies actually do. We touched on a few technical concepts such as MLFlow and Delta lake to create the right foundations for you to extend and adapt to specific requirements.
