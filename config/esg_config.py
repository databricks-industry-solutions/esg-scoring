# Databricks notebook source
# MAGIC %pip install PyPDF2==1.26.0 gensim==3.8.3 nltk==3.5 wordcloud==1.8.1 spacy==3.2.1

# COMMAND ----------

import re
useremail = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
username = useremail.split('@')[0]
username_sql_compatible = re.sub('\W', '_', username)

# COMMAND ----------

# Please replace this cell should you want to store data somewhere else.
database_name = f"esg_scoring_{username_sql_compatible}"
_ = sql("CREATE DATABASE IF NOT EXISTS {}".format(database_name))

# Similar to database, we will store actual content on a given path
data_path = f"/FileStore/{useremail}/esg_scoring"
dbutils.fs.mkdirs(data_path)

# Where we might stored temporary data on local disk
from pathlib import Path
temp_directory = f"/tmp/{username}/esg_scoring"
Path(temp_directory).mkdir(parents=True, exist_ok=True)

# COMMAND ----------

config = {
            "csr_fail_if_404"       : True,
            "csr_raw_table"         : f"{database_name}.csr_raw",
            "csr_raw_path"          : f"{data_path}/csr/raw",
            "csr_org_path"          : f"{data_path}/csr/organizations",
            "csr_org_table"         : f"{database_name}.csr_organizations",
            "csr_statements_table"  : f"{database_name}.csr_statements",
            "csr_statements_path"   : f"{data_path}/csr/statements",
            "csr_initiatives_table" : f"{database_name}.csr_initiatives",
            "csr_initiatives_path"  : f"{data_path}/csr/initiatives",
            "csr_scores_table"      : f"{database_name}.csr_scores",
            "csr_scores_path"       : f"{data_path}/csr/scores",
            "model_topic_name"      : f"esg_topics_{username_sql_compatible}",
            "model_scraper_name"    : f"esg_scraper_{username_sql_compatible}",
            "gdelt_raw_path"        : f"{data_path}/gdelt/raw",
            "gdelt_bronze_path"     : "/mnt/industry-gtm/fsi/solutions/esg_scoring/gdelt",
            "gdelt_silver_table"    : f"{database_name}.gdelt_silver",
            "gdelt_silver_path"     : f"{data_path}/gdelt/silver",
            "gdelt_gold_table"      : f"{database_name}.gdelt_gold",
            "gdelt_gold_path"       : f"{data_path}/gdelt/gold",
            "gdelt_scores_table"    : f"{database_name}.gdelt_score",
            "gdelt_scores_path"     : f"{data_path}/gdelt/score",
         }

# COMMAND ----------

import mlflow
experiment_name = f"/Users/{useremail}/esg_scoring_experiment"
mlflow.set_experiment(experiment_name) 

# COMMAND ----------

def tear_down():
  _ = sql("DROP DATABASE {} CASCADE".format(database_name))
  dbutils.fs.rm(data_path, True)
  import shutil
  shutil.rmtree(temp_directory)
