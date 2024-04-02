# Databricks notebook source
# MAGIC %md
# MAGIC # CSR scoring
# MAGIC In the absence of ESG standards, the onus falls on individual companies and investors to ensure high-fidelity ESG disclosures as well as to verify the sustainability of vendors, suppliers, customers, and counterparties. In this notebook, we will use natural language processing (NLP) techniques to identify common ESG themes and create a taxonomy that can be used by to compare organizations more objectively. 

# COMMAND ----------

# MAGIC %run ./config/configure_notebook

# COMMAND ----------

# MAGIC %md
# MAGIC ## Text preprocessing
# MAGIC
# MAGIC We apply [latent dirichlet allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) to learn topics descriptive to CSR reports. We want to be able to better understand and eventually summarize complex CSR reports into a specific ESG related themes. Before doing so, we need to further process our text content (converting words into their simplest grammatical forms) for NLP analysis.

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
from typing import Iterator
import pandas as pd
from utils.nlp_utils import *

@pandas_udf('string')
def lemmatize(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    load_nltk(nltk_path)
    for xs in batch_iter:
        yield xs.apply(lemmatize_text)

# COMMAND ----------

from pyspark.sql import functions as F
csr_df = spark.read.table(csr_table_statement).join(spark.read.table(portfolio_table), ['ticker'])
esg_df = csr_df.withColumn('lemma', lemmatize(F.col('statement')))
esg_df = esg_df.filter(F.length('lemma') > 255)
corpus = esg_df.select('lemma').toPandas().lemma

# COMMAND ----------

display(esg_df.select('ticker', 'sector', 'lemma'))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extracting topics
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

# np.random.RandomState was deprecated, so Hyperopt now uses np.random.Generator
import hyperopt
import numpy as np

if hyperopt.__version__.split('+')[0] > '0.2.5':
  rstate=np.random.default_rng(123)
else:
  rstate=np.random.RandomState(123)

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
spark_trials = SparkTrials(parallelism=num_executors)

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

best_params

# COMMAND ----------

corpus_B.unpersist(blocking=True)

# COMMAND ----------

# MAGIC %md
# MAGIC With multiple models trained in parallel, we can access our best set of hyperparameters that minimized our loss function set above (function of LDA perplexity). Although we used MLFlow to track multiple experiments, we did not log a physical model yet (we only tracked parameters and metrics). Given our set of best parameters, we train our model as a sklearn pipeline that contains our pre-processing steps (count vectorizer).

# COMMAND ----------

import mlflow
from sklearn.pipeline import make_pipeline

with mlflow.start_run(run_name=model_name) as run:

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
# MAGIC ## Interpreting results
# MAGIC We want to evaluate model relevance using more domain expertise. Would those topics make sense from an ESG perspective? Do we have clear categories defined across the Environmental, Social and Governance categories? By interacting with our model through simple visualization, we may want to name each topic into a specific policy in line with [GRI standards](https://www.globalreporting.org/standards/). Better, why not leveraging Generative AI capabilities to define our taxonomy? As an introductory solution, we report below a simple usage of DBRX model to name each topic we have discovered through our machine learning model. The logical extension of this solution would be to fine tune our foundational model against GRI standards. 

# COMMAND ----------

vocab = vectorizer.get_feature_names_out()
for topic, comp in enumerate(lda.components_): 
    word_idx = np.argsort(comp)[::-1][:100]
    print(f'******************* TOPIC {topic} *******************')
    print(' '.join([vocab[i] for i in word_idx]))

# COMMAND ----------

# MAGIC %md
# MAGIC We report the significant keywords describing each topic. These keywords will be brought as a context to a genAI model.

# COMMAND ----------

topic_keywords = []
vocab = vectorizer.get_feature_names_out()
for topic, comp in enumerate(lda.components_): 
    word_idx = np.argsort(comp)[::-1][:100]
    topic_keywords.append([vocab[i] for i in word_idx])

# COMMAND ----------

from langchain import FewShotPromptTemplate
from langchain import PromptTemplate

topics_prompt_tmpl = """###
[topic]: {topic_id}
[keywords]: {topic_keywords}
"""

prefix = """An analyst has been able to extract {num_topics} distinct topics from multiple corporate sustainability reports.
You are trying to find the best keyword that describes each topic. 
This taxonomy was expressed as a form of important keywords for each topic as represented below

"""

topic_prompt = PromptTemplate(
  input_variables=['topic_id', 'topic_keywords'],
  template=topics_prompt_tmpl
)

def build_topics_prompt(xs):
  topics = []
  for topic_id, topic_keywords in enumerate(xs):
    topics.append({'topic_id': topic_id, 'topic_keywords': ','.join(topic_keywords)})
  return topics

few_shot_prompt_template = FewShotPromptTemplate(
  examples=build_topics_prompt(topic_keywords),
  example_prompt=topic_prompt,
  prefix=prefix,
  suffix='',
  input_variables=['num_topics'],
  example_separator="\n"
)

system_prompt = few_shot_prompt_template.format(
  num_topics=int(best_params['n_components'])
).strip()

# COMMAND ----------

def query_gen_ai(system, user, temperature=0.1, max_tokens=200):
  chat_response = client.predict(
      endpoint="databricks-dbrx-instruct",
      inputs={
          "messages": [
              {
                "role": "system",
                "content": system_prompt
              },
              {
                "role": "user",
                "content": user_prompt
              }
          ],
          "temperature": temperature,
          "max_tokens": max_tokens
      }
  )
  return chat_response['choices'][-1]['message']['content']

# COMMAND ----------


import mlflow.deployments
client = mlflow.deployments.get_deploy_client("databricks")

def get_topic_description(user_prompt):
  return query_gen_ai(system_prompt, user_prompt)

def get_topic_name(response):
  import re
  m = re.search('\"(.*)\"', response)
  return m.group(1)

topic_names = []
for i in range(int(best_params['n_components'])):
  user_prompt = f"How would you name topic {i}?"
  topic_description = get_topic_description(user_prompt)
  topic_name = get_topic_name(topic_description)
  topic_names.append([i, topic_name, topic_description])

# COMMAND ----------

topic_df = pd.DataFrame(topic_names, columns=['id', 'policy', 'description'])
display(topic_df)

# COMMAND ----------

# MAGIC %md
# MAGIC For validation purpose, we represent each topic we extracted from our PDF document alongside topic title we learned from our DBRX model.

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
tf_feature_names = vectorizer.get_feature_names_out()
fig = plt.figure(figsize=(20, 20 * topics / 3))

# Display wordcloud for each extracted topic
for i, topic in enumerate(lda.components_):
    ax = fig.add_subplot(topics, 3, i + 1)
    ax.set_title(topic_names[i][1])
    wordcloud = word_cloud(lda, tf_feature_names, i)
    ax.imshow(wordcloud)
    ax.axis('off')

plt.savefig("/tmp/{}_wordcloud.png".format(model_name))

# COMMAND ----------

_ = (
  spark.createDataFrame(topic_df)
    .write
    .format('delta')
    .mode('overwrite')
    .saveAsTable(csr_table_topics)
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

_ = (
  gold_df
    .write
    .format('delta')
    .mode('overwrite')
    .saveAsTable(csr_table_gold)
)

# COMMAND ----------

_ = sql("OPTIMIZE {} ZORDER BY ticker".format(csr_table_gold))

# COMMAND ----------

esg_group = spark.read.table(csr_table_gold).toPandas()
esg_group = esg_group.merge(topic_df, on='id')[['organization', 'policy', 'probability']]
display(esg_group)

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

import mlflow

class EsgTopicAPI(mlflow.pyfunc.PythonModel):
    
    def __init__(self, pipeline):
        self.pipeline = pipeline  

    def load_context(self, context): 
        import nltk
        nltk.download('wordnet')
        nltk.download('punkt')
        nltk.download('omw-1.4')

    def _lemmatize(self, text):
        import nltk
        import re
        from nltk.stem import WordNetLemmatizer, PorterStemmer
        from utils.nlp_utils import tokenize
        results = []
        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()
        for token in tokenize(text):
            stem = stemmer.stem(lemmatizer.lemmatize(token))
            matcher = re.match('\w+', stem)
            if matcher:
                part = matcher.group(0)
                if len(part) > 3:
                    results.append(part)
        return ' '.join(results)

    def predict(self, context, series):
        lemma = series.apply(self._lemmatize)
        predictions = self.pipeline.transform(lemma)
        import pandas as pd
        return pd.Series([[float(p) for p in distribution] for distribution in predictions])

# COMMAND ----------

from mlflow.models import infer_signature

python_model = EsgTopicAPI(pipeline)
model_input = pd.Series(['''creat social impact woman board woman suit woman leader woman team member bipoc board bipoc suit bipoc leader bipoc team member team member veteran disabl lgbtq 378m spend with small divers supplier includ 190m spend with supplier certifi major bipoc black indigen peopl color woman veteran peopl with disabl commit nonprofit organ focus advanc social justic divers equiti equal inclus health equiti resili healthcar increas over 2020 environment disclosuressoci govern healthcar environment social'''])
model_output = python_model.predict(None, model_input)
model_signature = infer_signature(model_input, model_output)
model_signature

# COMMAND ----------

import sklearn
conda_env = mlflow.pyfunc.get_default_conda_env()
conda_env['dependencies'][2]['pip'] += ['scikit-learn=={}'.format(sklearn.__version__)]
conda_env['dependencies'][2]['pip'] += ['nltk=={}'.format(nltk.__version__)]
conda_env['dependencies'][2]['pip'] += ['pandas=={}'.format(pd.__version__)]
conda_env['dependencies'][2]['pip'] += ['numpy=={}'.format(np.__version__)]

# COMMAND ----------

with mlflow.start_run(run_name=model_name) as run:
    mlflow.pyfunc.log_model("model", 
                            python_model=python_model, 
                            signature=model_signature, 
                            pip_requirements=conda_env,
                            input_example=pd.DataFrame(model_input, columns=['data'])
                            )

# COMMAND ----------

from mlflow.tracking import MlflowClient

mlflow.set_registry_uri('databricks-uc')
client = MlflowClient()
latest_model = mlflow.register_model(f'runs:/{run.info.run_id}/model', model_registered_name)
client.set_registered_model_alias(model_registered_name, "production", latest_model.version)

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
    .table(csr_table_gold)
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
    .saveAsTable(csr_table_scores)
)

# COMMAND ----------

_ = sql("OPTIMIZE {} ZORDER BY (ticker)".format(csr_table_scores))

# COMMAND ----------

# MAGIC %md
# MAGIC We store our scores on a delta table that will be combined in our next notebook with news analytics and can be visualized as-is. What company talks the most (or is the most specific about their initiatives) across our set of machine learned policies? We represent companies ESG focus across the E, S and G using a simple bar chart.

# COMMAND ----------

esg_csr_data = ( 
  csr_scores_df
    .join(spark.createDataFrame(topic_df), ['id'])
    .groupBy('ticker', 'policy')
    .agg(F.avg('score').alias('score'))
    .toPandas()
    .pivot(index='ticker', columns='policy', values='score')
)

# COMMAND ----------

esg_csr_data['sum'] = esg_csr_data.sum(axis=1)
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


