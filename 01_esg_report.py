# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/esg-scoring on the `web-sync` branch. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/esg.

# COMMAND ----------

# MAGIC %md
# MAGIC # CSR reports
# MAGIC 
# MAGIC Any large scale organisation is now facing tremendous pressure from their shareholders to disclose more information about their environmental, social and governance strategies. Typically released on their websites on a yearly basis as a form of a PDF document, companies communicate on their key ESG initiatives across multiple themes such as how they value their employees, clients or customers, how they positively contribute back to society or even how they reduce (or commit to reduce) their carbon emissions. Consumed by third parties agencies, these reports are usually consolidated and benchmarked across industries to create ESG metrics. In this notebook, we would like to programmatically access 40+ ESG reports from top tier financial services institutions and learn key ESG initiatives across different topics. 

# COMMAND ----------

# MAGIC %run ./config/esg_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract PDF
# MAGIC In this section, we search for publicly available corporate sustainability documents from publicly traded organizations ([example](https://home.barclays/content/dam/home-barclays/documents/citizenship/ESG/Barclays-PLC-ESG-Report-2019.pdf)). Instead of going through each company website, one could access information from [responsibilityreports.com](https://www.responsibilityreports.com), manually browsing for content or automatically scraping new records (please check [T&Cs](https://www.responsibilityreports.com/Disclaimer)). For the purpose of this exercise, we only provide URLs to a few financial services organizations listed below. Please be aware that the quality of analytics derived in that solution strongly depends on the amount of PDF documents to learn from. We recommend training models with 100+ PDFs to extract meaningful insights (even if the end goal is only to classify a few documents).  

# COMMAND ----------

from io import StringIO
import pandas as pd

csv_str = """organisation,url
discover,https://www.responsibilityreports.com/Click/2357
equifax,https://www.responsibilityreports.com/Click/1346
canadian imperial bank,https://www.responsibilityreports.com/Click/1894
citigroup,https://www.responsibilityreports.com/Click/1515
eurobank,https://www.responsibilityreports.com/Click/3126
jpmorgan chase,https://www.responsibilityreports.com/Click/1278
keybank,https://www.responsibilityreports.com/Click/1599
laurentian bank of canada,https://www.responsibilityreports.com/Click/1918
national australia bank,https://www.responsibilityreports.com/Click/1555
pnc,https://www.responsibilityreports.com/Click/1829
standard chartered,https://www.responsibilityreports.com/Click/2781
tcf financial,https://www.responsibilityreports.com/Click/1669
wells fargo,https://www.responsibilityreports.com/Click/1904
ameriprise,https://www.responsibilityreports.com/Click/1784
lazard,https://www.responsibilityreports.com/Click/1429
capital one,https://www.responsibilityreports.com/Click/1640
goldman sachs,https://www.responsibilityreports.com/Click/1496"""

esg_pdf = pd.read_csv(StringIO(csv_str))
display(esg_pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC We download text content for online CSR disclosures using the `PyPDF2` library. Please note that you will need to support outbound HTTP access from your databricks workspace. Although having a central place where to source data from (such as [responsibilityreports.com](https://www.responsibilityreports.com)) minimizes the amount of firewall rules to enable, this approach comes at a price: it prevents user from distributing that scraping logic across multiple executors. Just like many web scraping processes, please proceed with extra caution and refer to responsibilityreports.com [T&Cs](https://www.responsibilityreports.com/Disclaimer) before doing so.

# COMMAND ----------

# MAGIC %run ./utils/esg_utils

# COMMAND ----------

esg_pdf['content'] = esg_pdf['url'].apply(get_csr_content)
display(esg_pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC PDFs are highly unstructured by nature with text that is often scattered across multiple lines, pages, columns. From a simple set of regular expressions to a more complex NLP model (we use a `NLTK` trained pipeline), we show how to extract clean sentences from raw text documents. Not defined as a silver bullet but more as a working example, feel free to extend this framework further.

# COMMAND ----------

esg_df = spark.createDataFrame(esg_pdf)
esg_df = esg_df.withColumn('statement', F.explode(extract_statements('content'))).drop('content').cache()
esg_df.write.mode('overwrite').format('delta').option('path', config['csr_raw_path']).saveAsTable(config['csr_raw_table'])

# COMMAND ----------

esg_df = spark.read.table(config['csr_raw_table'])
display(esg_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Organization contribution
# MAGIC Some organizations may be mentioned in the news via their common names (e.g. IBM) rather than their legal entities (e.g. International Business Machines Corporation), resulting in an overall poor match later when joining our datasets "as-is". Furthermore, many organizations are related to others, such as portfolio companies belonging to a same holding and mentioned in both public disclosure and news. Analysing ESG initiatives must take the global dynamics of a business into account. For that purpose, we will use NLP techniques to extract mentions of any organizations in a given CSR report and link those "named entities" to better capture ESG related news we know are relevant for any given organization.

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
from typing import Iterator
import pandas as pd

@pandas_udf('array<string>')
def extract_organizations_udf(content_series_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
  nlp = load_spacy()
  for content_series in content_series_iter:
    yield content_series.map(lambda text: extract_organizations(text, nlp))

# COMMAND ----------

org_df = esg_df.select(F.col('organisation'), F.explode(extract_organizations_udf(F.col('statement'))).alias('alias')).filter(F.length('alias') > 0)
org_df.write.format('delta').mode('overwrite').option('path', config['csr_org_path']).saveAsTable(config['csr_org_table'])
organisations = set(list(org_df.toPandas().alias.apply(clean_org_name)) + list(org_df.toPandas().organisation))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Learn topics
# MAGIC 
# MAGIC In this section, we apply [latent dirichlet allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) to learn topics descriptive to CSR reports. Please note that the goal is to set the right foundations with a baseline model that can be further extended with different NLP techniques. We want to be able to better understand and eventually sumarize complex CSR reports into a specific ESG related themes (such as 'valuing employees'). Before doing so, we may want to consider additional stopwords that you may want to change to accomodate different industries or domain specific language

# COMMAND ----------

import gensim
from gensim.parsing.preprocessing import STOPWORDS

# add company names as stop words
org_stop_words = []
for organisation in organisations:
    for t in organisation.split(' '):
        org_stop_words.append(t)

# our list contains all english stop words + companies names + specific keywords
stop_words = STOPWORDS.union(org_stop_words)

# COMMAND ----------

esg_df = esg_df.withColumn('lemma', lemmatize_text('statement')).filter(F.length('lemma') > 255)
corpus = esg_df.select('lemma').toPandas().lemma

# COMMAND ----------

# MAGIC %md
# MAGIC The challenge of topic modelling is to extract good quality of topics that are clear and meaningful. This depends heavily on the quality of text preprocessing (above), the amount of data to learn from and the strategy of finding the optimal number of topics. With more data (more PDFs), we may learn more meaningful insights. With industry specific ESG reports, we may learn industry specific ESG initiatives as opposition to broader catagories. We highly recommend starting small with the following code snippet and further extend this framework with more data / more specific data accordingly. Although we could use `hyperopt` to tune parameters of a [LDA](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) pipeline, we wanted to explicitly set the number of topics to 6 in the context of that solution accelerator, "lock" that experiment to a define set of topics we could learn more and interpret.

# COMMAND ----------

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words = stop_words, ngram_range = (1,1))
vec_model = vectorizer.fit(corpus)

# COMMAND ----------

import mlflow
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import LatentDirichletAllocation

with mlflow.start_run(run_name='esg_lda') as run:

  lda = LatentDirichletAllocation(
    n_components=6,
    max_iter=150,
    evaluate_every=3,
    random_state=42,
    verbose=False
  )
  
  # train pipeline
  pipeline = make_pipeline(vec_model, lda)
  pipeline.fit(corpus)

  # log model
  mlflow.sklearn.log_model(pipeline, 'pipeline')
  
  # Mlflow run ID
  lda_run_id = mlflow.active_run().info.run_id

# COMMAND ----------

# MAGIC %md
# MAGIC ### Interpreting results
# MAGIC We want to evaluate model relevance using more domain expertise. Would those topics make sense from an ESG perspective? Do we have clear categories defined spanning accross the Environmental, Social and Governance broader categories? By interacting with our model through simple visualizations, we want to name each topic into a specific policy in line with [GRI standards](https://www.globalreporting.org/standards/).

# COMMAND ----------

import matplotlib.pyplot as plt
from wordcloud import WordCloud

# We ensure relevance of our topics using simple wordcloud visualisation
def word_cloud(model, tf_feature_names, index):
    
    imp_words_topic=""
    comp = model.components_[index]
    tfs = ['_'.join(t.split(' ')) for t in tf_feature_names]
    vocab_comp = zip(tfs, comp)
    sorted_words = sorted(vocab_comp, key = lambda x:x[1], reverse=True)[:200]
    
    for word in sorted_words:
        imp_words_topic = imp_words_topic + " " + word[0]
    
    return WordCloud(
        background_color="white",
        width=300, 
        height=300, 
        contour_width=2, 
        contour_color='steelblue'
    ).generate(imp_words_topic)
    
topics = len(lda.components_)
tf_feature_names = vectorizer.get_feature_names()
fig = plt.figure(figsize=(20, 20 * topics / 3))

# Display wordcloud for each extracted topic
for i, topic in enumerate(lda.components_):
    ax = fig.add_subplot(topics, 3, i + 1)
    wordcloud = word_cloud(lda, tf_feature_names, i)
    ax.imshow(wordcloud)
    ax.axis('off')

plt.savefig(f"{temp_directory}/wordcloud.png")

# COMMAND ----------

# MAGIC %md
# MAGIC For different PDF documents, you may have to look at wordcloud visualizations above and rename topics accordingly. As mentioned, different data will yield different insights, and more documents to learn from may result in more specific themes / categories. We leave interpretation of those topics to our users with the ability to map those to internal policies or external guidelines / regulations. We can infer topic distribution for each statement that we store to a delta table. For each statement, we retrieve the most descriptive topic and its meaning given our nomenclature defined above. 

# COMMAND ----------

from typing import Iterator
from pyspark.sql.functions import pandas_udf
lda_run_id_B = sc.broadcast(lda_run_id)

@pandas_udf("array<float>")
def describe_topics(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
  import mlflow
  model = mlflow.sklearn.load_model("runs:/{}/pipeline".format(lda_run_id_B.value))
  for batch in batch_iter:
    predictions = model.transform(batch)
    yield pd.Series([[float(p) for p in distribution] for distribution in predictions])

# COMMAND ----------

esg_df \
  .withColumn('probabilities', describe_topics('lemma')) \
  .drop('lemma') \
  .write \
  .format('delta') \
  .mode('overwrite') \
  .option('path', config['csr_statements_path']) \
  .saveAsTable(config['csr_statements_table'])

# COMMAND ----------

display(spark.read.table(config['csr_statements_table']))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Retrieve initiatives
# MAGIC Using a partitioning window, we extract the most descriptive policies for each organization. Although a few policies seems to be properly categorized, we can observe some obvious misclassification. At a first glance, we observe statements containing metrics to be often "recognized" as environmental related. Given the quantifiable nature of environmental policies, our model may be biased towards the use of number, percent signs or metrics. However, despite some misclassification (remember that we used topic modelling to discover unknown themes rather than classifying known labels), we show how one could dramatically simplify a complex PDF document of hundreds of pages into specific initiatives, answering questions like "*What did company X do with regards to environmental policy?*"

# COMMAND ----------

from pyspark.sql.window import Window
import numpy as np

@F.udf('int')
def get_topic(xs):
  return int(np.argmax(xs))

spark \
  .read \
  .table(config['csr_statements_table']) \
  .withColumn('id', get_topic('probabilities')) \
  .withColumn('probability', F.array_max('probabilities')) \
  .withColumn('rank', F.row_number().over(Window.partitionBy('organisation', 'id').orderBy(F.desc('probability')))) \
  .select('organisation','url', 'statement', 'rank', 'probability', 'id') \
  .write \
  .format('delta') \
  .mode('overwrite') \
  .option('path', config['csr_initiatives_path']) \
  .saveAsTable(config['csr_initiatives_table'])

# COMMAND ----------

display(
  spark
    .read
    .table(config['csr_initiatives_table'])
    .withColumnRenamed('id', 'topic_id')
    .filter(F.col('rank') == 1)
    .filter(F.col('probability') > 0.8)
    .orderBy(F.desc('probability'))
    .select('organisation', 'topic_id', 'statement')
)

# COMMAND ----------

# MAGIC %md
# MAGIC As our framework was built around the use of AI, the themes we learned from will be consistent across every organisations. In addition to summarizing complex PDF documents, this ML driven framework can be used to quantify the un-quantifiable and objectively compare non metrics initiatives across organisations, answering questions like "*How much more does company X focus on the wellbeing of their employees compare to company Y?*". For that purpose, we create a simple pivot table that will summarize companies' focus across our machine learned policies.

# COMMAND ----------

esg_group = spark.read.table(config['csr_statements_table']).toPandas()
esg_group['topics'] = esg_group['probabilities'].apply(lambda xs: [[i, x] for i, x in enumerate(xs)])
esg_group = esg_group.explode('topics').reset_index(drop=True)
esg_group['topic_id'] = esg_group['topics'].apply(lambda x: x[0])
esg_group['probability'] = esg_group['topics'].apply(lambda x: x[1])
esg_group = esg_group[['organisation', 'topic_id', 'probability']]
esg_group = pd.pivot_table(
  esg_group, 
  values='probability', 
  index='organisation',
  columns=['topic_id'], 
  aggfunc=np.sum)

# COMMAND ----------

# scale topic frequency between 0 and 1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))
esg_focus = pd.DataFrame(scaler.fit_transform(esg_group), columns=esg_group.columns)
esg_focus.index = esg_group.index

# plot heatmap, showing main area of focus for each company across topics we learned
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(rc={'figure.figsize':(12,8)})
sns.heatmap(esg_focus, annot=False, cmap='Greens')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, we were able to create a framework that helps financial analysts objectively assess the sustainable impact of their investments, retailers to compare the ethical posture of their suppliers or organisations to compare their environmental initiatives with their closest competitors.

# COMMAND ----------

# MAGIC %md
# MAGIC ## CSR score
# MAGIC In the previous section, we set the foundations to a AI driven ESG framework by learning key ESG initiatives as opposition to broad statements. By looking at how descriptive each statement is, we create a simple score by rank ordering organisations. This score will be the building block to our next notebook where we will be able to quantify how much a company talks about ESG vs. how much they walk the talk. 

# COMMAND ----------

_ = (
  spark
    .read
    .table(config['csr_statements_table'])
    .withColumn('probabilities', with_topic('probabilities'))
    .withColumn('probabilities', F.explode('probabilities'))
    .select(
      F.col('probabilities.id'),
      F.col('organisation'),
      F.col('probabilities.probability')
    )
    .groupBy('id', 'organisation')
    .agg(
      F.sum('probability').alias('esg')
    )
    .withColumn('rank', F.row_number().over(Window.partitionBy('id').orderBy('esg')))
    .withColumn('esg', F.round(F.col('rank') * 100 / F.lit(esg_pdf.shape[0])))
    .withColumnRenamed('id', 'topic_id')
    .select('organisation', 'topic_id', 'esg')
    .write
    .format('delta')
    .mode('overwrite')
    .option('path', config['csr_scores_path'])
    .saveAsTable(config['csr_scores_table'])
)

# COMMAND ----------

esg_csr_data = ( 
  spark
    .read
    .table(config['csr_scores_table'])
    .withColumnRenamed('id', 'topic_id')
    .groupBy('organisation')
    .agg(F.avg('esg').alias('esg'))
    .toPandas()
)

esg_csr_data['sum'] = esg_csr_data.sum(axis=1)
esg_csr_data.index = esg_csr_data['organisation']
esg_csr_data = esg_csr_data.sort_values(by='sum', ascending=False).drop('sum',  axis=1)
esg_csr_data.plot.bar(
  rot=90, 
  stacked=False, 
  title='ESG score based on corporate disclosures',
  ylabel='ESG score',
  ylim=[0, 100],
  figsize=(16, 8)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Actionable models
# MAGIC Now that we've built our model, we may want to port it out to classify different documents or transfer the intelligence we learned to a different data feed such as news articles. However, we built a few pipelines and text preparation (such as lemmatization) that would need to be shipped alongside the LDA model itself. This data preparation can be embedded as part of a `pyfunc` model as follows. Our approach is to wrap our lemmatization process as data preparation for topic classification and ensure each python dependency required will be explicitly embedded with the relevant version to run independently.

# COMMAND ----------

class EsgTopicAPI(mlflow.pyfunc.PythonModel):
    
  def __init__(self, pipeline):
    self.pipeline = pipeline  
    
  def load_context(self, context): 
    import nltk
    nltk.data.path.append(context.artifacts['wordnet'])

  def _lemmatize(self, text):
    import nltk
    from nltk.stem import WordNetLemmatizer, PorterStemmer
    from gensim.utils import simple_preprocess
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    results = [stemmer.stem(lemmatizer.lemmatize(token)) for token in simple_preprocess(text)]
    return ' '.join(results)

  def predict(self, context, series):
    lemma = series.apply(self._lemmatize)
    predictions = pipeline.transform(lemma)
    import pandas as pd
    return pd.Series([[float(p) for p in distribution] for distribution in predictions])

# COMMAND ----------

import sklearn

with mlflow.start_run(run_name='esg_topic_classification'):

  conda_env = mlflow.pyfunc.get_default_conda_env()
  conda_env['dependencies'][2]['pip'] += ['scikit-learn=={}'.format(sklearn.__version__)]
  conda_env['dependencies'][2]['pip'] += ['gensim=={}'.format(gensim.__version__)]
  conda_env['dependencies'][2]['pip'] += ['nltk=={}'.format(nltk.__version__)]
  conda_env['dependencies'][2]['pip'] += ['pandas=={}'.format(pd.__version__)]
  conda_env['dependencies'][2]['pip'] += ['numpy=={}'.format(np.__version__)]
  
  wordnet_dir = "/dbfs{}/wordnet".format(data_path)
  nltk.download('wordnet', download_dir=wordnet_dir)
  artifacts = {
    'wordnet': wordnet_dir
  }
  
  mlflow.pyfunc.log_model(
    'pipeline', 
    python_model=EsgTopicAPI(pipeline), 
    conda_env=conda_env,
    artifacts=artifacts
  )
  
  api_run_id = mlflow.active_run().info.run_id
  print(api_run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC By registering our model to ML registry, we make it available to downstream processes and backend jobs. We will simply load our `pyfunc` logic as a simple mlflow import.

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
model_uri = "runs:/{}/pipeline".format(api_run_id)
result = mlflow.register_model(model_uri, config['model_topic_name'])
version = result.version

# COMMAND ----------

# MAGIC %md
# MAGIC We can also promote our model to different stages programmatically. Although our models would need to be reviewed in real life scenario, we make it available as a production artifact for our next notebook focused on news analytics as well as archiving any previous iteration

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
for model in client.search_model_versions("name='{}'".format(config['model_topic_name'])):
  if model.current_stage == 'Production':
    print("Archiving model version {}".format(model.version))
    client.transition_model_version_stage(
      name=config['model_topic_name'],
      version=int(model.version),
      stage="Archived"
    )

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name=config['model_topic_name'],
    version=version,
    stage="Production"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Take away
# MAGIC In this first section, we set the foundations to a data driven ESG framework. We've demonstrated how AI can be used to extract key ESG initiatives from unstructured PDF documents and use this intelligence to create a more objective way to quantify ESG strategies from public companies. With the vocabulary we have learned using topic modelling, we can re-use that model to see how much of these initiatives were actually followed through and what was the media reception using news analytics data.
