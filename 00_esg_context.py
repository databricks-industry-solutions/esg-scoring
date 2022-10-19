# Databricks notebook source
# MAGIC %md
# MAGIC <img src=https://d1r5llqwmkrl74.cloudfront.net/notebooks/fs-lakehouse-logo.png width="600px">
# MAGIC 
# MAGIC [![COMPLEXITY](https://img.shields.io/badge/COMPLEXITY-201-orange)]()
# MAGIC [![POC](https://img.shields.io/badge/POC-5d-red)]()
# MAGIC 
# MAGIC 
# MAGIC *The future of finance goes hand in hand with social responsibility, environmental stewardship and corporate ethics. In order to stay competitive, Financial Services Institutions (FSI)  are increasingly  disclosing more information about their environmental, social and governance (ESG) performance. By better understanding and quantifying the sustainability and societal impact of any investment in a company or business, FSIs can mitigate reputation risk and maintain the trust with both their clients and shareholders. At Databricks, we increasingly hear from our customers that ESG has become a C-suite priority. This is not solely driven by altruism but also by economics: [Higher ESG ratings are generally positively correlated with valuation and profitability while negatively correlated with volatility](https://corpgov.law.harvard.edu/2020/01/14/esg-matters/). In this solution, we offer a novel approach to sustainable finance by combining NLP techniques and news analytics to extract key strategic ESG initiatives and learn companies' commitments to corporate responsibility.*
# MAGIC 
# MAGIC 
# MAGIC ___
# MAGIC <antoine.amend@databricks.com>

# COMMAND ----------

# MAGIC %md
# MAGIC <img src=https://d1r5llqwmkrl74.cloudfront.net/notebooks/fsi/esg_scoring/images/reference_architecture.png width="800px">

# COMMAND ----------

# MAGIC %run ./config/esg_config

# COMMAND ----------

# Initialize the environment
tear_down()

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | PyPDF2                                 | PDF parser              | BSD        | https://pypi.org/project/PyPDF2                     |
# MAGIC | spark-gdelt                            | GDELT wrapper           | Apache2    | https://github.com/aamend/spark-gdelt               |
# MAGIC | PyLDAvis                               | LDA visualizer          | MIT        | https://github.com/bmabey/pyLDAvis                  |
# MAGIC | Gensim                                 | Topic modelling         | L-GPL2     | https://radimrehurek.com/gensim/                    |
# MAGIC | Wordcloud                              | Visualization library   | MIT        | https://github.com/amueller/word_cloud              |
# MAGIC | fuzzywuzzy                             | Fuzzy matching library  | GNU        | https://github.com/seatgeek/fuzzywuzzy              |
# MAGIC | python-Levenshtein                     | Fuzzy matching library  | GNU        | https://github.com/ztane/python-Levenshtein         |
