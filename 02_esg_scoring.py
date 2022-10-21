# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/esg-scoring on the `web-sync` branch. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/esg.

# COMMAND ----------

# MAGIC %md
# MAGIC # News analytics
# MAGIC 
# MAGIC As covered in the previous section, we were able to compare businesses side by side across different ESG initiatives. Although we created a simple ESG score, we want our score **not to be subjective but truly data driven**. In other terms, we do not want to solely base our assumptions on companies’ official disclosures but rather on how companies' reputations are perceived in the media, across all 3 environmental, social and governance variables. For that purpose, we will be using [GDELT](https://www.gdeltproject.org/), the global database of event location and tones as a building block to that framework.

# COMMAND ----------

# MAGIC %run ./config/esg_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrieve news
# MAGIC *Supported by Google Jigsaw, the [GDELT](https://www.gdeltproject.org/) Project monitors the world's broadcast, print, and web news from nearly every corner of every country in over 100 languages and identifies the people, locations, organizations, themes, sources, emotions, counts, quotes, images and events driving our global society every second of every day, creating a free open platform for computing on the entire world.* Although it is convenient to scrape for [master URL]((http://data.gdeltproject.org/gdeltv2/lastupdate.txt) file to process latest GDELT increment, processing 2 years backlog is time consuming and resource intensive (please **proceed with caution**). Below script is for illustration purpose only on a small time window. 

# COMMAND ----------

from datetime import datetime
from datetime import timedelta
from datetime import date

max_date = datetime.today()
min_date = max_date - timedelta(hours=1)

# COMMAND ----------

import urllib.request
master_url = 'http://data.gdeltproject.org/gdeltv2/masterfilelist.txt'
master_file = urllib.request.urlopen(master_url)
to_download = []
for line in master_file:
  decoded_line = line.decode("utf-8")
  if 'gkg.csv.zip' in decoded_line:
    a = decoded_line.split(' ')
    file_url = a[2].strip()
    file_dte = datetime_object = datetime.strptime(file_url.split('/')[-1].split('.')[0], '%Y%m%d%H%M%S')
    if (file_dte > min_date and file_dte <= max_date):
      to_download.append(file_url)

print("{} file(s) to download from {} to {}".format(len(to_download), min_date, max_date))

# COMMAND ----------

# MAGIC %md
# MAGIC Unfortunately, GDELT does only offer single files packed as a `zip` archive. Zip being an archive rather than a compressed file, it cannot be read programmatically using native python or spark. As part of this download, we also convert `zip` to `gzip` format that we safely store to distributed file storage.

# COMMAND ----------

from zipfile import ZipFile
import gzip
import io
import os

def download_content(url, save_path):
  with urllib.request.urlopen(url) as dl_file:
    input_zip = ZipFile(io.BytesIO(dl_file.read()), "r")
    name = input_zip.namelist()[0]
    with gzip.open(save_path, 'wb') as f:
      f.write(input_zip.read(name))

def download_to_dbfs(url):
  file_name = '{}.gz'.format(url.split('/')[-1][:-4])
  tmp_file = '{}/{}'.format(temp_directory, file_name)
  download_content(url, tmp_file)
  dbutils.fs.mv('file:{}'.format(tmp_file), 'dbfs:{}/{}'.format(config['gdelt_raw_path'], file_name))
    
n = len(to_download)
for i, url in enumerate(to_download):
  download_to_dbfs(url)
  print("{}/{} [{}]".format(i + 1, n, url))

# COMMAND ----------

display(dbutils.fs.ls(config['gdelt_raw_path']))

# COMMAND ----------

# MAGIC %md
# MAGIC The structure of GDELT GKGv2 files is complex (see [data model](http://data.gdeltproject.org/documentation/GDELT-Global_Knowledge_Graph_Codebook-V2.1.pdf)). Because of its complex taxonomy, some fields may be separated by comma, spaces and nested entities separated by hash or semi columns. We make use of a scala library to access schematized records as a clean dataframe rather than handling this business logic ourselves. 

# COMMAND ----------

# We pass some configuration parameters from python to scala via the spark configuration
spark.conf.set('gdelt.raw.path', config['gdelt_raw_path'])
spark.conf.set('gdelt.bronze.path', config['gdelt_bronze_path'])

# COMMAND ----------

# MAGIC %scala
# MAGIC import com.aamend.spark.gdelt._
# MAGIC display(spark.read.gdeltGkgV2(spark.conf.get("gdelt.raw.path")))

# COMMAND ----------

# MAGIC %md
# MAGIC With enough content downloaded, one can easily create an external table pointing to a directory where gdelt files have been schematized and validated. For the purpose of this exercise, we will be using data previously loaded to a given directory.
# MAGIC 
# MAGIC ```
# MAGIC 
# MAGIC import com.aamend.spark.gdelt._
# MAGIC 
# MAGIC spark
# MAGIC   .read
# MAGIC   .gdeltGkgV2(spark.conf.get("gdelt.raw.path"))
# MAGIC   .write
# MAGIC   .format("delta")
# MAGIC   .mode("append")
# MAGIC   .save(spark.conf.get("gdelt.bronze.path"))
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC Given the volume of data available in GDELT (100 million records for the last 18 months only), we leverage the [lakehouse](https://databricks.com/blog/2020/01/30/what-is-a-data-lakehouse.html) paradigm by moving data from raw, to filtered and enriched, respectively from Bronze, Silver and Gold layers, and can easily extend our process to operate in near real time (GDELT files are published every 15mn) using [Auto Loader](https://docs.databricks.com/spark/latest/structured-streaming/auto-loader.html) functionality. 

# COMMAND ----------

from pyspark.sql import functions as F

gdelt_df = (
  spark
    .read
    .format('delta')
    .load(config['gdelt_bronze_path'])
    .filter(F.col('gkgRecordId.translingual') == False)
)

display(
  gdelt_df
    .withColumn('date', F.to_date('publishDate'))
    .groupBy('date')
    .count()
    .orderBy(F.asc('date'))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrieve organizations news coverage
# MAGIC We now have access to news analytics records for every single organisation worldwide, from local news event to major world breaking news. Although we do not have access to the underlying HTML article, GDELT created a multiple of metadata we could use as-is (such as organisation mention, location, themes, etc.)

# COMMAND ----------

gdelt_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC Although GDELT does not redistribute the HTML content of each article, it provides us with the article title as an extra XML tag that we retrieve using a simple user defined function

# COMMAND ----------

from pyspark.sql.functions import udf
import re

@udf('string')
def extract_title(xml):
  if xml:
    m = re.search('<PAGE_TITLE>(.+?)</PAGE_TITLE>', xml)
    if m:
      return str(m.group(1))
  return ''

# COMMAND ----------

# MAGIC %md
# MAGIC In order to limit the number of articles, we can easily pre-filter our data for specific themes. See [documentation](http://data.gdeltproject.org/documentation/GCAM-MASTER-CODEBOOK.TXT) for a list of all themes supported by GDELT project. As we only are interested for articles around ESG, we access content classified as "UNGP" (united nation guiding principles) that we can further extend to a broader set of themes.

# COMMAND ----------

@udf('boolean')
def esg_theme(xs):
  for x in xs:
    if x.startswith('UNGP') or x.startswith("ENV"):
      return True
  return False

# COMMAND ----------

# MAGIC %md
# MAGIC ### Organization contribution
# MAGIC Some organizations may be mentioned in the news via their common names (e.g. IBM) rather than their legal entities (e.g. International Business Machines Corporation), resulting in an overall poor match when joining our datasets "as-is". Furthermore, many organizations are related to others, such as portfolio companies belonging to a same holding. Analysing ESG initiatives must take the global dynamics of a business into account. We will be using all mentions of organizations found in coprorate disclosures from our previous NLP notebook.

# COMMAND ----------

alias_df = (
  spark
    .read
    .table(config['csr_org_table'])
    .groupBy('organisation', 'alias')
    .count()
    .withColumnRenamed('count', 'tf')
)

csr_contribution = (
  alias_df
    .groupBy('organisation')
    .count()
    .join(alias_df, ['organisation'])
    .withColumn('csr_contribution', F.col('tf') / F.col('count'))
    .drop('count', 'tf')
)

display(csr_contribution)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Media contribution
# MAGIC As most news articles would be mentioning more than one organization, we need a similar weighting strategy. Furthermore, many businesses (especially in the media industry) might be overindexed, resulting in irrelevant insights and probably low ESG scores. We use a simple term frequency analysis (TF-IDF) to understand the significance of each organization to a given article and attribute its impact proortionally. 

# COMMAND ----------

# MAGIC %run ./utils/esg_utils

# COMMAND ----------

@udf('string')
def clean_org_name_udf(text):
  return clean_org_name(text)

# COMMAND ----------

# how many times was organization A mentioned in that article
gdelt_org_tf = (
  gdelt_df
    .withColumn('organisation', F.explode('organisations'))
    .withColumn('organisation', clean_org_name_udf('organisation'))
    .groupBy(F.col('organisation'), F.col('documentIdentifier'))
    .count()
    .withColumnRenamed('count', 'tf')
)

# how many articles mentioned organization A
gdelt_org_df = gdelt_org_tf.groupBy('organisation').count().withColumnRenamed('count', 'df')

# COMMAND ----------

# significance of organization A for a given article
gdelt_n = gdelt_org_tf.select('documentIdentifier').distinct().count()
gdelt_org_tfidf = (
  gdelt_org_tf.join(gdelt_org_df, ['organisation'])
    .select(
      F.col('organisation'),
      F.col('documentIdentifier'),
      (F.col('df') * F.log(F.lit(gdelt_n) / F.col('df'))).alias('tfidf')
    )
)

# normalization
gdelt_contribution = (
  gdelt_org_tfidf
    .groupBy('documentIdentifier')
    .agg(F.sum('tfidf').alias('sum_tfidf'))
    .join(gdelt_org_tfidf, ['documentIdentifier'])
    .select(
      F.col('documentIdentifier'),
      F.col('organisation').alias('alias'),
      (F.col('tfidf') / F.col('sum_tfidf')).alias('gdelt_contribution')
    )
)

# COMMAND ----------

esg_contribution = gdelt_contribution.join(csr_contribution, ['alias']).cache()
display(esg_contribution)

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, we have built a basic weighted indicator that would help us assess ESG coverage for any indicator proportional to its significance. The more targeted an article is towards a specific organisation, higher is its ESG contribution. This process can be summarized in the workflow below.

# COMMAND ----------

# MAGIC %md
# MAGIC <img src=https://d1r5llqwmkrl74.cloudfront.net/notebooks/fsi/esg_scoring/images/news_contribution.png width="300px">

# COMMAND ----------

gdelt_silver_df = (
  gdelt_df
    .join(esg_contribution, ['documentIdentifier'])
    .withColumn('title', extract_title('extrasXML'))
    .filter(F.length(F.col('title')) > 100)
    .withColumn('contribution', F.col('gdelt_contribution') * F.col('csr_contribution'))
    .select(
      F.to_date('publishDate').alias('date'),
      F.col('organisation'),
      F.col('sourceCommonName').alias('source'),
      F.col('documentIdentifier').alias('url'),
      F.col('title'),
      F.col('tone.tone').alias('tone'),
      F.col('contribution')
    )
)

_ = (
  gdelt_silver_df
    .write
    .format('delta')
    .mode('overwrite')
    .option('path', config['gdelt_silver_path'])
    .saveAsTable(config['gdelt_silver_table'])
)

# COMMAND ----------

# MAGIC %md
# MAGIC Given our test dataset (a dozen of companies) and a couple of months worth of GDELT history, we already have hundreds of thousands records to extract ESG insights from and can represent news events as a media coverage timeline. The visualization below offers a lot of insights on its own as a way to monitor daily coverage as new events unfold. 

# COMMAND ----------

from pyspark.sql import functions as F
display(
  spark
    .read
    .table(config['gdelt_silver_table'])
    .groupBy('date', 'organisation')
    .count()
    .orderBy('date', 'organisation')
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classify news
# MAGIC Although GDELT provide us with a lot of content already, we can take an extra step and scrape the actual article content. We could create a user defined function to scrape HTML site, parse and clean content. Luckily, our GDELT library `com.aamend.spark:spark-gdelt` already takes care of that process by leveraging cloud parallelism (distributed webscraper) as follows. However, for the purpose of that solution, and because many corporate users may not have internet connectivity from their cloud environment to 60,000 possible news sources, we focus on article title that is already part of GDELT metadata. 

# COMMAND ----------

# MAGIC %md
# MAGIC ```scala
# MAGIC import com.aamend.spark.gdelt.ContentFetcher
# MAGIC 
# MAGIC val contentFetcher = new ContentFetcher()
# MAGIC   .setInputCol("sourceUrl")
# MAGIC   .setOutputTitleCol("title")
# MAGIC   .setOutputContentCol("content")
# MAGIC   .setOutputKeywordsCol("keywords")
# MAGIC   .setOutputPublishDateCol("publishDate")
# MAGIC   .setOutputDescriptionCol("description")
# MAGIC   .setUserAgent("Mozilla/5.0 (X11; U; Linux x86_64; de; rv:1.9.2.8) Gecko/20100723 Ubuntu/10.04 (lucid) Firefox/3.6.8")
# MAGIC   .setConnectionTimeout(1000)
# MAGIC   .setSocketTimeout(1000)
# MAGIC 
# MAGIC val contentDF = contentFetcher.transform(gdeltEventDS)
# MAGIC contentDF.show()
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC In this section, we retrieve the model we trained in our previous stage and apply its predictive value to news title in order to describe news articles into consistent categories.

# COMMAND ----------

def load_model():
  import mlflow
  return mlflow.pyfunc.load_model("models:/{}/production".format(config['model_topic_name']))

# COMMAND ----------

import pandas as pd
from typing import Iterator
from pyspark.sql.functions import pandas_udf

@pandas_udf("array<float>")
def classify(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
  model = load_model()
  for batch in batch_iter:
    yield model.predict(batch)

# COMMAND ----------

gkg_classified = (
  spark
    .read
    .table(config['gdelt_silver_table'])
    .withColumn('probabilities', classify('title'))
    .withColumn('probabilities', with_topic('probabilities'))
    .select('date', 'organisation', 'source', 'url', 'title', 'probabilities', 'tone', 'contribution')
)

_ = (
  gkg_classified
    .write
    .format('delta')
    .mode('overwrite')
    .option('path', config['gdelt_gold_path'])
    .saveAsTable(config['gdelt_gold_table'])
)

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, we have been able to describe news information using the insights we learned from corporate responsibility reports. Only setting the foundation, we highly recommend user to train a model with more data to learn more specific ESG categories and access underlying HTML articles content as previously described. We also recommend user tagging each of those machine learned topic against internal policies or external regulations / guidelines.

# COMMAND ----------

display(spark.read.table(config['gdelt_gold_table']))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ESG score
# MAGIC In the previous section, we showed how the intel we learned from CSR reports could be transfered into the world of news to describe any article against a set of ESG policies. Using sentiment analysis (tone is part of the GDELT metadata), we aim at detecting how much more "positive" or "negative" a company is perceived across those machine learned policies. We apply our weighted factor to score organizations more objectively (taking into account news coverage or portfolio companies)

# COMMAND ----------

from pyspark.sql import functions as F

esg_raw = (
  spark
    .read
    .table(config['gdelt_gold_table'])
    .withColumn('probability', F.explode('probabilities'))
    .select(
      F.col('organisation'),
      F.col('probability.id').alias('topic_id'),
      (F.col('probability.probability') * F.col('contribution')).alias('weighted_count'),
      (F.col('probability.probability') * F.col('tone') * F.col('contribution')).alias('weighted_tone'),
    )
    .groupBy('organisation', 'topic_id')
    .agg((F.sum('weighted_tone') / F.sum('weighted_count')).alias('esg'))
    .toPandas()
)

# COMMAND ----------

# MAGIC %md
# MAGIC Similar to our previous notebook, we can represent that ESG coverage against each of our machine learned topics. *How much more positive is organisation A covered in the news for ESG related events that organisation B?*

# COMMAND ----------

import numpy as np
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))

esg_group = pd.pivot_table(
  esg_raw, 
  values='esg', 
  index='organisation',
  columns=['topic_id'], 
  aggfunc=np.mean)
 
esg_focus = pd.DataFrame(scaler.fit_transform(esg_group), columns=esg_group.columns)
esg_focus.index = esg_group.index
 
# plot heatmap, showing main area of focus for each company across topics we learned
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(rc={'figure.figsize':(12,8)})
sns.heatmap(esg_focus, annot=False, cmap='Blues')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC We normalize our dataset to rank organisations by how much coverage and how much descriptive each news article is, storing our insights onto a delta table.

# COMMAND ----------

from scipy.stats import percentileofscore

esg_scores = esg_raw.groupby(['topic_id'])['esg'].agg(list)
esg_scores_norm = pd.DataFrame(esg_scores).rename({'esg': 'esg_dist'}, axis=1)
esg_norm = esg_raw.merge(esg_scores_norm, left_on=['topic_id'], right_on=['topic_id'])

def norm(score, dist):
  return percentileofscore(dist, score)

esg_norm['esg'] = esg_norm.apply(lambda x: norm(x.esg, x.esg_dist), axis=1)

esg_norm = esg_norm[['topic_id', 'organisation', 'esg']]

spark \
  .createDataFrame(esg_norm) \
  .write \
  .format('delta') \
  .mode('overwrite') \
  .option('path', config['gdelt_scores_path']) \
  .saveAsTable(config['gdelt_scores_table'])

# COMMAND ----------

# MAGIC %md
# MAGIC Similar to our previous section, we can represent organisations ESG media coverage as a simple bar chart. 

# COMMAND ----------

esg_gdelt_data = ( 
  spark
    .read
    .table(config['gdelt_scores_table'])
    .withColumnRenamed('id', 'topic_id')
    .groupBy('organisation')
    .agg(F.avg('esg').alias('esg'))
    .toPandas()
)

esg_gdelt_data['sum'] = esg_gdelt_data.sum(axis=1)
esg_gdelt_data.index = esg_gdelt_data['organisation']
esg_gdelt_data = esg_gdelt_data.sort_values(by='sum', ascending=False).drop('sum',  axis=1)
esg_gdelt_data.plot.bar(
  rot=90, 
  stacked=False, 
  title='ESG score based on news analytics',
  ylabel='ESG score',
  ylim=[0, 100],
  figsize=(16, 8)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Walking the talk
# MAGIC Finally, we can combine our insights generated from our 2 notebooks to get a sense of how much each companies' initiative were followed through and what was the media coverage (positive or negative). Given our objective and data driven approach (rather than subjective scoring going through a PDF document), we can detect organisations "walking the talk". 

# COMMAND ----------

from pyspark.sql import functions as F

esg_walk = (
  spark
    .read
    .table(config['gdelt_scores_table'])
    .withColumnRenamed('esg', 'walk')
    .select('organisation', 'topic_id', 'walk')
)

esg_talk = (
  spark
    .read
    .table(config['csr_scores_table'])
    .withColumnRenamed('esg', 'talk')
    .select('organisation', 'topic_id', 'talk')
)

display(
  esg_walk
    .join(esg_talk, ['organisation', 'topic_id'])
    .withColumn('walkTheTalk', F.col('walk') - F.col('talk'))
    .orderBy(F.desc('walkTheTalk'))
)

# COMMAND ----------

# MAGIC %md
# MAGIC On the left hand side, we see organizations consistently scoring higher using news analytics than CSR reports. Those companies may not disclose a lot of information as part of their yearly disclosures (and arguably may have a poor ESG score from rating agencies) but consistently do good. Their support to communities or environmental impact is noticed and positive. On the right hand side come organisations disclosing more than actually doing or organisations constantly mentionned negatively in the press. Although we report a confidence level (poor, medium, high) based on media coverage, this solution accelerator is only providing you with the foundations required to adopt a data driven ESG framework rather than actual insights.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Take away
# MAGIC Throughout this series of notebooks, we introduced a novel approach to environmental, social and governance to objective quantify the ESG impact of public organisations using AI. By combining corporate disclosure and news analytics data, we demonstrated how machine learning could be used to bridge the gap between what companies say and what companies actually do. We touched on a few technical concepts such as MLFlow and Delta lake to create the right foundations for you to extend and adapt to specific requirements. 
