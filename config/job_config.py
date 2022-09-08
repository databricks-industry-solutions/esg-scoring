from dbacademy.dbgems import get_cloud
def get_job_param_json(env, solacc_path, job_name, node_type_id, spark_version, spark):
    num_workers = 8
    job_json = {
        "timeout_seconds": 36000,
        "name": job_name,
        "max_concurrent_runs": 1,
        "tags": {
            "usage": "solacc_testing",
            "group": "FSI"
        },
        "tasks": [
            {
                "job_cluster_key": "esg_cluster",
                "notebook_task": {
                    "notebook_path": f"{solacc_path}/00_esg_context"
                },
                "task_key": "00_esg_context",
                "description": ""
            },
            {
                "job_cluster_key": "esg_cluster",
                "notebook_task": {
                    "notebook_path": f"{solacc_path}/01_esg_report"
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
                    "notebook_path": f"{solacc_path}/02_esg_scoring",
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
                    "spark_version": spark_version,
                "spark_conf": {
                    "spark.databricks.delta.formatCheck.enabled": "false"
                    },
                    "num_workers": num_workers,
                    "node_type_id": node_type_id,
                    "custom_tags": {
                      "usage": "solacc_testing"
                    }
                }
            }
        ]
    }
    cloud = get_cloud()
    if cloud == "AWS": 
      job_json["job_clusters"][0]["new_cluster"]["aws_attributes"] = {
                        "ebs_volume_count": 0,
                        "availability": "ON_DEMAND",
                        "instance_profile_arn": "arn:aws:iam::997819012307:instance-profile/shard-demo-s3-access",
                        "first_on_demand": 1
                    }
    if cloud == "MSA": 
      job_json["job_clusters"][0]["new_cluster"]["azure_attributes"] = {
                        "availability": "ON_DEMAND_AZURE",
                        "first_on_demand": 1
                    }
    if cloud == "GCP": 
      job_json["job_clusters"][0]["new_cluster"]["gcp_attributes"] = {
                        "use_preemptible_executors": False
                    }
    return job_json

    