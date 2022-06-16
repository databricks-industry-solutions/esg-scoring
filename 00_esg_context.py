# Databricks notebook source
# MAGIC %md
# MAGIC <img src=https://d1r5llqwmkrl74.cloudfront.net/notebooks/fsi/fs-lakehouse-logo-transparent.png width="600px">
# MAGIC 
# MAGIC [![DBR](https://img.shields.io/badge/DBR-9.1ML-red?logo=databricks&style=for-the-badge)](.)
# MAGIC [![CLOUD](https://img.shields.io/badge/CLOUD-n1--highmem--16-blue?logo=googlecloud&style=for-the-badge)]()
# MAGIC [![POC](https://img.shields.io/badge/POC-5 days-green?style=for-the-badge)]()
# MAGIC 
# MAGIC *The future of finance goes hand in hand with social responsibility, environmental stewardship and corporate ethics. 
# MAGIC In order to stay competitive, Financial Services Institutions (FSI)  are increasingly  disclosing more information 
# MAGIC about their environmental, social and governance (ESG) performance. By better understanding and quantifying the 
# MAGIC sustainability and societal impact of any investment in a company or business, FSIs can mitigate reputation risk and 
# MAGIC maintain the trust with both their clients and shareholders. At Databricks, we increasingly hear from our customers 
# MAGIC that ESG has become a C-suite priority. This is not solely driven by altruism but also by economics: 
# MAGIC Higher ESG ratings are generally positively correlated with valuation and profitability while negatively correlated with 
# MAGIC volatility ([source](https://corpgov.law.harvard.edu/2020/01/14/esg-matters/)). 
# MAGIC In this solution, we offer a novel approach to sustainable finance by combining NLP techniques and news analytics to 
# MAGIC extract key strategic ESG initiatives and learn companies' commitments to corporate responsibility.*
# MAGIC 
# MAGIC ___
# MAGIC 
# MAGIC <antoine.amend@databricks.com>

# COMMAND ----------

# MAGIC %md
# MAGIC <img src='https://d1r5llqwmkrl74.cloudfront.net/notebooks/fsi/esg_scoring/gcp/images/reference_architecture.png' width=800>

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | beautifulsoup4                         | Web scraper library     | MIT        | https://www.crummy.com/software/BeautifulSoup       |
# MAGIC | PyPDF2                                 | PDF parser              | BSD        | https://pypi.org/project/PyPDF2                     |
# MAGIC | NLTK                                   | NLP toolkit             | Apache2    | https://github.com/nltk/nltk                        |
# MAGIC | Spacy                                  | NLP toolkit             | MIT        | https://spacy.io/                                   |
# MAGIC | Wordcloud                              | Visualization library   | MIT        | https://github.com/amueller/word_cloud              |
