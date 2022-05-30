# Databricks notebook source
# MAGIC %md
# MAGIC # ESG scores
# MAGIC Because many corporate users may not have internet connectivity from their cloud environment to 60,000 possible news sources, we focus on article title that is already part of GDELT metadata rather than scraping the actual HTML content. In this section, we retrieve the model we trained in our previous stage and apply its predictive value to news titles in order to describe news articles into consistent ESG categories.

# COMMAND ----------

# MAGIC %run ./config/configure_notebook

# COMMAND ----------

gdelt_bronze_table = config['database']['tables']['gdelt']['bronze']
gdelt_silver_table = config['database']['tables']['gdelt']['silver']
gdelt_scores_table = config['database']['tables']['gdelt']['scores']

csr_topics_table = config['database']['tables']['csr']['topics']
csr_bronze_table = config['database']['tables']['csr']['bronze']
csr_scores_table = config['database']['tables']['csr']['scores']

portfolio_table = config['database']['tables']['portfolio']

# COMMAND ----------

gdelt_df = spark.read.table(gdelt_bronze_table)
display(gdelt_df)

# COMMAND ----------

def load_model():
  import mlflow
  model_name = config['model']['tagger']['name']
  return mlflow.pyfunc.load_model("models:/{}/staging".format(model_name))

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

from pyspark.sql import functions as F
from utils.spark_utils import *
 
_ = (
  gdelt_df
    .withColumn('probabilities', classify('title'))
    .withColumn('probabilities', with_topic('probabilities'))
    .withColumn('relevance', F.col('org_contribution') * F.col('gkg_contribution'))
    .select('publishDate', 'ticker', 'source', 'url', 'title', 'probabilities', 'tone', 'relevance')
    .write
    .format('delta')
    .mode('overwrite')
    .saveAsTable(gdelt_silver_table)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ESG coverage
# MAGIC We were able to describe news information using the insights we learned from corporate responsibility reports. Only setting the foundation, we highly recommend user to train a model with more data to learn more specific ESG categories and access underlying HTML articles content as previously described. Similar to our previous notebook, we can represent the media / sentiment coverage for each organisation. How much more negative, or positive each organisation is mentioned in the news against our ESG policies?

# COMMAND ----------

esg_weighted = (
  spark
    .read
    .table(gdelt_silver_table)
    .withColumn('topic', F.explode('probabilities'))
    .withColumn('weightedTone', F.col('topic.probability') * F.col('relevance') * F.col('tone'))
    .withColumn('weightedCoverage', F.col('relevance') * F.col('topic.probability'))
    .groupBy('ticker', 'topic.id')
    .agg(
      F.sum('weightedTone').alias('weightedTone'),
      F.sum('weightedCoverage').alias('weightedCoverage'),
      F.sum(F.col('topic.probability') * F.col('relevance')).alias('weightedToneNorm'),
      F.sum('relevance').alias('weightedCoverageNorm')
    )
    .withColumn('esgTone', F.col('weightedTone') / F.col('weightedToneNorm'))
    .withColumn('esgCoverage', F.col('weightedCoverage') / F.col('weightedCoverageNorm'))
)

# COMMAND ----------

import numpy as np
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))
topics_df = spark.read.table(csr_topics_table)
organizations_df = spark.read.table(csr_bronze_table).select('ticker', 'organization')

# COMMAND ----------

esg_group = esg_weighted.join(topics_df, ['id']).join(organizations_df, ['ticker']).toPandas()
esg_group = pd.pivot_table(
  esg_group, 
  values='esgTone', 
  index='organization',
  columns=['policy'], 
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
# MAGIC ## ESG scores
# MAGIC We showed how the intelligence we have learned from CSR reports could be transfered into the world of news to better describe any article against a set of ESG policies. Using sentiment analysis (tone is part of the GDELT metadata), we aim at detecting how much more "positive" or "negative" a company is perceived across those machine learned policies. We create a score internal to each company across its 'E', 'S' and 'G' dimensions.

# COMMAND ----------

from pyspark.sql.window import Window
 
orgs = esg_weighted.select('ticker').distinct().count()
 
_ = (
  esg_weighted
    .withColumn('score', F.row_number().over(Window.partitionBy('id').orderBy('esgTone')))
    .withColumn('score', F.col('score') * F.lit(100) / F.lit(orgs))
    .write
    .format('delta')
    .mode('overwrite')
    .saveAsTable(gdelt_scores_table)
)

# COMMAND ----------

esg_gdelt_data = ( 
  spark
    .read
    .table(gdelt_scores_table)
    .join(topics_df, ['id'])
    .join(organizations_df, ['ticker'])
    .groupBy('organization', 'topic')
    .agg(F.avg('score').alias('score'))
    .toPandas()
    .pivot(index='organization', columns='topic', values='score')
)
 
esg_gdelt_data['sum'] = esg_gdelt_data.sum(axis=1)
esg_gdelt_data = esg_gdelt_data.sort_values(by='sum', ascending=False).drop('sum',  axis=1)
esg_gdelt_data.plot.bar(
  rot=90, 
  stacked=False, 
  color={"E": "#A1D6AF", "S": "#D3A1D6", "G": "#A1BCD6"},
  title='ESG score based on media coverage',
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
 
esg_walk = spark.read.table(gdelt_scores_table).withColumnRenamed('score', 'walk')
esg_talk = spark.read.table(csr_scores_table).withColumnRenamed('score', 'talk')
 
display(
  esg_walk
    .join(esg_talk, ['ticker', 'id'])
    .join(topics_df, ['id'])
    .withColumn('walkTheTalk', F.col('walk') - F.col('talk'))
    .orderBy(F.desc('walkTheTalk'))
    .select('ticker', 'topic', 'policy', 'walk', 'talk')
)

# COMMAND ----------

# MAGIC %md
# MAGIC On the left hand side, we see organizations consistently scoring higher using news analytics than CSR reports. Those companies may not disclose a lot of information as part of their yearly disclosures (and arguably may have a poor ESG score from rating agencies) but consistently do good. Their support to communities or environmental impact is noticed and positive. On the right hand side come organisations disclosing more than actually doing or organisations constantly mentionned negatively in the press. However, you may notice that we did not take coverage bias into account here. Organisations mentioned more frequently than others may tend to have a more negative score due to the negative nature of news analytics. We leave this as an open thought for future solution. In spite of this caveat, this solution accelerator is providing you with the foundations required to adopt a data driven ESG framework that you could further extend.

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, we persist all of our organizations that we could have scored through this solution, having both a CSR and a GDELT scores.

# COMMAND ----------

_ = (
  esg_walk
    .join(esg_talk, ['ticker', 'id'])
    .join(spark.read.table(csr_bronze_table), ['ticker'])
    .select('organization', 'sector', 'ticker')
    .distinct()
    .write
    .format('delta')
    .saveAsTable(portfolio_table)
)

# COMMAND ----------

_ = sql('OPTIMIZE {} ZORDER BY ticker, organization'.format(portfolio_table))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Take Away
# MAGIC Throughout this series of notebooks, we introduced a novel approach to environmental, social and governance to objective quantify the ESG impact of public organisations using AI. By combining corporate disclosure and news analytics data, we demonstrated how machine learning could be used to bridge the gap between what companies say and what companies actually do. We touched on a few technical concepts such as MLFlow and Delta lake to create the right foundations for you to extend and adapt to specific requirements.
