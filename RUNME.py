# Databricks notebook source
# MAGIC %md This notebook sets up the companion cluster(s) to run the solution accelerator. It also creates the Workflow to create a Workflow DAG and illustrate the order of execution. Feel free to interactively run notebooks with the cluster or to run the Workflow to see how this solution accelerator executes. Happy exploring!
# MAGIC 
# MAGIC The pipelines, workflows and clusters created in this script are not user-specific, so if another user alters the workflow and cluster via the UI, running this script again after the modification resets them.
# MAGIC 
# MAGIC **Note**: If the job execution fails, please confirm that you have set up other environment dependencies as specified in the accelerator notebooks. Accelerators sometimes require the user to set up additional cloud infra or data access, for instance. 

# COMMAND ----------

# DBTITLE 0,Install util packages
# MAGIC %pip install git+https://github.com/databricks-academy/dbacademy@v1.0.13 git+https://github.com/databricks-industry-solutions/notebook-solution-companion@safe-print-html --quiet --disable-pip-version-check

# COMMAND ----------

from solacc.companion import NotebookSolutionCompanion

# COMMAND ----------

job_json = {
        "timeout_seconds": 36000,
        "max_concurrent_runs": 1,
        "tags": {
            "usage": "solacc_testing",
            "group": "FSI"
        },
        "tasks": [
            {
                "job_cluster_key": "esg_cluster",
                "notebook_task": {
                    "notebook_path": f"00_esg_context"
                },
                "task_key": "00_esg_context",
                "description": ""
            },
            {
                "job_cluster_key": "esg_cluster",
                "notebook_task": {
                    "notebook_path": f"01_esg_report"
                },
                "task_key": "01_esg_report",
                "depends_on": [
                    {
                        "task_key": "00_esg_context"
                    }
                ],
                "description": ""
            },
            {
                "job_cluster_key": "esg_cluster",
                "notebook_task": {
                    "notebook_path": f"02_esg_scoring",
                },
                "task_key": "02_esg_scoring",
                "libraries": [
                  {
                    "maven": {
                      "coordinates": "com.aamend.spark:spark-gdelt:3.0"
                    }
                  }
                ],
                "depends_on": [
                    {
                        "task_key": "01_esg_report"
                    }
                ]
            }
        ],
        "job_clusters": [
            {
                "job_cluster_key": "esg_cluster",
                "new_cluster": {
                    "spark_version": "10.4.x-cpu-ml-scala2.12",
                "spark_conf": {
                    "spark.databricks.delta.formatCheck.enabled": "false"
                    },
                    "num_workers": 8,
                    "node_type_id": {"AWS": "i3.xlarge", "MSA": "Standard_DS3_v2", "GCP": "n1-highmem-4"},
                    "custom_tags": {
                      "usage": "solacc_testing"
                    }
                }
            }
        ]
    }

# COMMAND ----------

dbutils.widgets.dropdown("run_job", "False", ["True", "False"])
run_job = dbutils.widgets.get("run_job") == "True"
NotebookSolutionCompanion().deploy_compute(job_json, run_job=run_job)

# COMMAND ----------


