# Databricks notebook source
# MAGIC %md
# MAGIC # CSR scoring
# MAGIC In the absence of ESG standards, the onus falls on individual companies and investors to ensure high-fidelity ESG disclosures as well as to verify the sustainability of vendors, suppliers, customers, and counterparties. In this notebook, we will use natural language processing (NLP) techniques to identify common ESG themes and create a taxonomy that can be used by to compare organizations more objectively. 

# COMMAND ----------

# MAGIC %run ./config/configure_notebook

# COMMAND ----------

model_name = config['model']['tagger']['name']
csr_gold   = config['database']['tables']['csr']['gold']
csr_bronze = config['database']['tables']['csr']['bronze']
csr_silver = config['database']['tables']['csr']['silver']
csr_topics = config['database']['tables']['csr']['topics']
csr_scores = config['database']['tables']['csr']['scores']

# COMMAND ----------

# MAGIC %md
# MAGIC ## Text preprocessing
# MAGIC 
# MAGIC We apply [latent dirichlet allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) to learn topics descriptive to CSR reports. We want to be able to better understand and eventually sumarize complex CSR reports into a specific ESG related themes. Before doing so, we need to further process our text content (converting words into their simplest grammatical forms) for NLP analysis.

# COMMAND ----------

from utils.nlp_utils import *

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
from typing import Iterator
import pandas as pd

@pandas_udf('string')
def lemmatize_text_udf(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
  load_nltk(nltk_path)
  for xs in batch_iter:
    yield xs.apply(lemmatize_text)

# COMMAND ----------

from pyspark.sql import functions as F
csr_df = spark.read.table(csr_silver)
esg_df = csr_df.withColumn('lemma', lemmatize_text_udf(F.col('statement')))
esg_df = esg_df.filter(F.length('lemma') > 255)
corpus = esg_df.select('lemma').toPandas().lemma

# COMMAND ----------

display(esg_df.select('organization', 'sector', 'statement'))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Hyperparameter tuning
# MAGIC The challenge of topic modelling is to extract good quality of topics that are clear and meaningful. This depends heavily on the quality of text preprocessing (above), the amount of data to learn from and the strategy of finding the optimal number of topics (below). With more data (more PDFs), we may learn more meaningful insights. With industry specific ESG reports, we may learn industry specific ESG initiatives as opposition to broader catagories. We highly recommend starting small with the following code snippet and further extend this framework with more data / more specific data accordingly. In the cell below, we will be using `hyperopts` to tune parameters of a [LDA](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) model.

# COMMAND ----------

# read default stopwords
with open('config/stopwords.txt', 'r') as f:
  stop_words = f.read().split('\n')
  
# consider organisations names as stop words
organizations = list(csr_df.select('organization').distinct().toPandas().organization)  
stop_words = get_stopwords(stop_words, organizations, nltk_path)

# COMMAND ----------

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words = stop_words, ngram_range = (1,1))
vec_model = vectorizer.fit(corpus)

# COMMAND ----------

corpus_B = sc.broadcast(corpus)

# COMMAND ----------

from sklearn.decomposition import LatentDirichletAllocation
from hyperopt import fmin, hp, tpe, SparkTrials, STATUS_OK
import numpy as np

def train_model(params):
  
  # define our LDA parameters
  lda = LatentDirichletAllocation(
    n_components=int(params['n_components']),
    learning_method='online',
    learning_decay=float(params['learning_decay']),
    max_iter=150,
    evaluate_every=3,
    random_state=42, 
    verbose=True
  )
  
  # train a model
  X = vec_model.transform(corpus_B.value)
  lda.fit(X)

  # minimizing perplexity at each step
  loss = lda.perplexity(X)
  return {'status': STATUS_OK, 'loss': loss}

# COMMAND ----------

# grid search our optimal number of topics
search_space = {
  'n_components': hp.quniform('n_components', 5, 10, 1),
  'learning_decay': hp.quniform('learning_decay', 0.65, 0.9, 0.05),
}

# we define the number of executors we have at our disposal
spark_trials = SparkTrials(parallelism=config['environment']['executors'])

# we retrieve the set of parameters that minimize our loss function
best_params = fmin(
  fn=train_model, 
  space=search_space, 
  algo=tpe.suggest, 
  max_evals=50, 
  trials=spark_trials, 
  rstate=rstate
)

# COMMAND ----------

corpus_B.unpersist(blocking=True)

# COMMAND ----------

# MAGIC %md
# MAGIC With multiple models trained in parallel, we can access our best set of hyperparameters that minimized our loss function set above (function of LDA perplexity). Although we used MLFlow to track multiple experiments, we did not log a physical model yet (we only tracked parameters and metrics). Given our set of best parameters, we train our model as a sklearn pipeline that contains our pre-processing steps (count vectorizer).

# COMMAND ----------

import mlflow
from sklearn.pipeline import make_pipeline

with mlflow.start_run(run_name='esg_lda') as run:

  lda = LatentDirichletAllocation(
    n_components=int(best_params['n_components']),
    learning_decay=float(best_params['learning_decay']),
    learning_method='online',
    max_iter=150,
    evaluate_every=3,
    random_state=42,
    verbose=True
  )
  
  mlflow.log_param("n_components", best_params['n_components'])
  mlflow.log_param("learning_decay", best_params['learning_decay'])
  
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
# MAGIC We want to evaluate model relevance using more domain expertise. Would those topics make sense from an ESG perspective? Do we have clear categories defined spanning accross the Environmental, Social and Governance broader categories? By interacting with our model through simple visualization, we want to name each topic into a specific policy in line with [GRI standards](https://www.globalreporting.org/standards/).

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

plt.savefig("/tmp/{}_wordcloud.png".format(model_name))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Name ESG topics
# MAGIC Although we were able to extract well defined topics that can describe initiatives in our CSR dataset, topics are meaningless unless overlayed with domain expertise. Please note that this naming convention below is only valid in the context of that experiment. In different settings, with different data, our model will yield different results and a different naming convention will apply.

# COMMAND ----------

import pandas as pd

topic_df = pd.DataFrame([
  [0, 'S', 'valuing employee'],
  [1, 'G', 'code of conduct'],
  [2, 'G', 'board of directors'],
  [3, 'G', 'risk management'],
  [4, 'S', 'supporting communities'],
  [5, 'E', 'energy transition']
], columns=['id', 'topic', 'policy'])

# COMMAND ----------

_ = (
  spark.createDataFrame(topic_df)
    .write
    .format('delta')
    .mode('overwrite')
    .saveAsTable(csr_topics)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Comparing organisations
# MAGIC As our framework was built with AI first, the themes we learned from will be consistent across every organisations. In addition to summarizing complex PDF documents, such a framework can be used to objectively compare non metrics initiatives across organisations, answering questions like "*How much more does company X focus on the wellbeing of their employees compare to company Y?*". 

# COMMAND ----------

from typing import Iterator
from pyspark.sql.functions import pandas_udf
from utils.spark_utils import *

lda_run_id_B = sc.broadcast(lda_run_id)

@pandas_udf("array<float>")
def describe_topics(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
  import mlflow
  model = mlflow.sklearn.load_model("runs:/{}/pipeline".format(lda_run_id_B.value))
  for batch in batch_iter:
    predictions = model.transform(batch)
    yield pd.Series([[float(p) for p in distribution] for distribution in predictions])

# COMMAND ----------

gold_df = (
  esg_df
    .withColumn('probabilities', describe_topics('lemma'))
    .withColumn('probabilities', with_topic(F.col('probabilities')))
    .withColumn('probability', F.explode(F.col('probabilities')))
    .withColumn('id', F.col('probability.id'))
    .withColumn('probability', F.col('probability.probability'))
    .drop('probabilities')
    .drop('lemma')
)

# COMMAND ----------

_ = sql("OPTIMIZE {} ZORDER BY ticker".format(csr_gold))

# COMMAND ----------

esg_group = spark.read.table(csr_gold).filter(F.col('ticker').isin(portfolio)).toPandas()
esg_group = esg_group.merge(topic_df, on='id')[['organization', 'policy', 'probability']]

# COMMAND ----------

# Probabilities give the attribution of a sentence to a given topic
# Aggregating those would show the weighted coverage in a company disclosure
# The more significant and the more frequent an initiative is drives higher score
esg_group = pd.pivot_table(
  esg_group, 
  values='probability', 
  index='organization',
  columns=['policy'], 
  aggfunc=np.sum)

# COMMAND ----------

# MAGIC %md
# MAGIC Using seaborn visualisation, we can easily flag key differences across our companies. When some organisations would put more focus on valuing employees and promoting diversity and inclusion, some seem to be more focused towards environmental initiatives. 

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

plt.savefig("/tmp/{}_heatmap.png".format(model_name))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Registering model
# MAGIC Now that we've built our model, we may want to port it out to classify different documents or transfer the intelligence we learned to a different data feed such as news articles. However, we built a few pipelines and text preparation (such as lemmatization) that would need to be shipped alongside the LDA model itself. This data preparation can be embedded as part of a `pyfunc` model as follows. Our approach is to wrap our lemmatization process as data preparation for topic classification and ensure each python dependency required will be explicitly embedded with the relevant version to run independently.

# COMMAND ----------

class EsgTopicAPI(mlflow.pyfunc.PythonModel):
    
  def __init__(self, pipeline):
    self.pipeline = pipeline  
    
  def load_context(self, context): 
    import nltk
    nltk.download('wordnet')

  def _lemmatize(self, text):
    import nltk
    import re
    from nltk.stem import WordNetLemmatizer, PorterStemmer
    from gensim.utils import simple_preprocess
    results = []
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    for token in simple_preprocess(text):
      stem = stemmer.stem(lemmatizer.lemmatize(token))
      matcher = re.match('\w+', stem)
      if matcher:
        part = matcher.group(0)
        if len(part) > 3:
          results.append(part)
    return ' '.join(results)

  def predict(self, context, series):
    lemma = series.apply(self._lemmatize)
    predictions = pipeline.transform(lemma)
    import pandas as pd
    return pd.Series([[float(p) for p in distribution] for distribution in predictions])

# COMMAND ----------

import sklearn

with mlflow.start_run(run_name=model_name):

  conda_env = mlflow.pyfunc.get_default_conda_env()
  conda_env['dependencies'][2]['pip'] += ['scikit-learn=={}'.format(sklearn.__version__)]
  conda_env['dependencies'][2]['pip'] += ['gensim=={}'.format(gensim.__version__)]
  conda_env['dependencies'][2]['pip'] += ['nltk=={}'.format(nltk.__version__)]
  conda_env['dependencies'][2]['pip'] += ['pandas=={}'.format(pd.__version__)]
  conda_env['dependencies'][2]['pip'] += ['numpy=={}'.format(np.__version__)]
  
  mlflow.pyfunc.log_model(
    'pipeline', 
    python_model=EsgTopicAPI(pipeline), 
    conda_env=conda_env
  )
  
  api_run_id = mlflow.active_run().info.run_id
  print(api_run_id)

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
client.log_artifact(api_run_id, "/tmp/{}_heatmap.png".format(model_name))
client.log_artifact(api_run_id, "/tmp/{}_wordcloud.png".format(model_name))
model_uri = 'runs:/{}/pipeline'.format(api_run_id)
result = mlflow.register_model(model_uri, model_name)
version = result.version

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
for model in client.search_model_versions("name='{}'".format(model_name)):
  if model.current_stage == 'Staging':
    print("Archiving model version {}".format(model.version))
    client.transition_model_version_stage(
      name=model_name,
      version=int(model.version),
      stage="Archived"
    )

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name=model_name,
    version=version,
    stage='Staging'
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## CSR score
# MAGIC In the previous section, we set the foundations to a AI driven ESG framework by learning key ESG initiatives as opposition to broad statements. By looking at how descriptive each statement is, we create a simple score by rank ordering organisations. This score will be the building block to our next notebook where we will be able to quantify how much a company talks about ESG vs. how much they walk the talk.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

num_orgs = sc.broadcast(len(organizations))

csr_scores_df = (
  spark
    .read
    .table(csr_gold)
    .groupBy('id', 'ticker', 'organization')
    .agg(F.sum('probability').alias('esg'))
    .withColumn('rank', F.row_number().over(Window.partitionBy('id').orderBy('esg')))
    .withColumn('score', F.round(F.col('rank') * 100 / F.lit(num_orgs.value)))
    .select('ticker', 'id', 'score')
)

# COMMAND ----------

_ = (
  csr_scores_df
    .write
    .format('delta')
    .mode('overwrite')
    .saveAsTable(csr_scores)
)

# COMMAND ----------

_ = sql("OPTIMIZE {} ZORDER BY (ticker)".format(csr_scores))

# COMMAND ----------

# MAGIC %md
# MAGIC We store our scores on a delta table that will be combined in our next notebook with news analytics and can be visualized as-is. What company talks the most (or is the most specific about their initiatives) across our set of machine learned policies? We represent companies ESG focus across the E, S and G using a simple bar chart.

# COMMAND ----------

organizations_df = spark.read.table(csr_bronze).select('ticker', 'organization')
esg_csr_data = ( 
  csr_scores_df
    .join(spark.createDataFrame(topic_df), ['id'])
    .join(organizations_df, ['ticker'])
    .filter(F.col('ticker').isin(portfolio))
    .groupBy('organization', 'topic')
    .agg(F.avg('score').alias('score'))
    .toPandas()
    .pivot(index='organization', columns='topic', values='score')
)

# COMMAND ----------

esg_csr_data['sum'] = esg_csr_data.sum(axis=1)
esg_csr_data = esg_csr_data.sort_values(by='sum', ascending=False).drop('sum',  axis=1)
esg_csr_data.plot.bar(
  rot=90, 
  stacked=False, 
  color={"E": "#A1D6AF", "S": "#D3A1D6", "G": "#A1BCD6"},
  title='ESG score based on corporate disclosures',
  ylabel='ESG score',
  ylim=[0, 100],
  figsize=(16, 8)
)
