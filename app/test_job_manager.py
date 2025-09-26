# # Databricks notebook source

# # COMMAND ----------
# # MAGIC %md
# ### This notebook tests the DBXMetaGenJobManager functionality
# ###
# ### Features:
# ### - Creates user-specific job names to avoid conflicts
# ### - Searches for and deletes existing jobs before creating new ones
# ### - Monitors job progress with detailed status updates
# ### - Stops execution immediately if job fails
# ### - Provides failure debugging information

# # TODO: delete this notebook

# # COMMAND ----------

# # MAGIC %pip install -r requirements.txt
# dbutils.library.restartPython()

# # COMMAND ----------

# from databricks.sdk import WorkspaceClient
# import yaml
# from job_manager import DBXMetaGenJobManager
# import time
# import sys

# w = WorkspaceClient()
# job_manager = DBXMetaGenJobManager(w)

# # Get current user to create unique job name
# current_user = w.current_user.me()
# username = current_user.user_name.split("@")[0]  # Get username part before @

# base_job_name = "test_dbxmetagen_job"
# job_name = f"{username}_{base_job_name}"

# tables = "dbxmetagen.default.finance_test"
# cluster_size = "Small (1-2 workers)"
# config_path = "./variables.yml"
# user_email = current_user.user_name

# with open(config_path, "r") as f:
#     test_config = yaml.safe_load(f)

# print(f"Looking for existing job with name: {job_name}")

# # Check if job already exists
# existing_jobs = w.jobs.list(name=job_name)
# existing_job_id = None

# for job in existing_jobs:
#     if job.settings.name == job_name:
#         existing_job_id = job.job_id
#         print(f"Found existing job with ID: {existing_job_id}")
#         break

# if existing_job_id:
#     print(f"Deleting existing job: {existing_job_id}")
#     try:
#         w.jobs.delete(existing_job_id)
#         print("✅ Existing job deleted successfully")
#     except Exception as e:
#         print(f"⚠️ Warning: Could not delete existing job: {e}")
#         print("Continuing with job creation...")
# else:
#     print("No existing job found")

# print(f"Creating new job: {job_name}")
# job_id, run_id = job_manager.create_metadata_job(
#     job_name=job_name,
#     tables=tables,
#     cluster_size=cluster_size,
#     config=test_config,
#     user_email=user_email,
# )

# print(f"Job created with ID: {job_id}")
# print(f"Run started with ID: {run_id}")
# print("Polling for job completion (checking every 30 seconds)...")

# # Terminal states for Databricks job runs
# terminal_states = {"SUCCESS", "FAILED", "CANCELED", "TIMEOUT"}
# failure_states = {"FAILED", "CANCELED", "TIMEOUT"}

# run_status = "PENDING"
# max_wait_time = 3600  # 1 hour timeout
# start_time = time.time()
# poll_count = 0

# while run_status not in terminal_states:
#     # Check for timeout
#     elapsed_time = time.time() - start_time
#     if elapsed_time > max_wait_time:
#         print(f"❌ Timeout reached after {max_wait_time/60:.1f} minutes")
#         print(f"Final status: {run_status}")
#         break

#     # Get current status
#     run_status = job_manager.get_run_status(run_id)
#     poll_count += 1

#     print(
#         f"Poll #{poll_count} - Status: {run_status} (elapsed: {elapsed_time/60:.1f} min)"
#     )

#     # If not terminal, wait before next poll
#     if run_status not in terminal_states:
#         time.sleep(30)

# # Final result
# print("\n" + "=" * 50)
# if run_status == "SUCCESS":
#     print("✅ Job completed successfully!")
#     final_result = "SUCCESS"
# elif run_status in failure_states:
#     print(f"❌ Job failed with status: {run_status}")
#     final_result = "FAILED"

#     # Get run details for debugging
#     try:
#         run_details = w.jobs.get_run(run_id)
#         if run_details.state and run_details.state.state_message:
#             print(f"Failure reason: {run_details.state.state_message}")

#         # Print logs URL if available
#         if run_details.run_page_url:
#             print(f"View run details: {run_details.run_page_url}")
#     except Exception as e:
#         print(f"Could not get run details: {e}")

#     print(f"Final status: {run_status}")
#     print(f"Total polling time: {(time.time() - start_time)/60:.1f} minutes")
#     print(f"Result: {final_result}")
#     print("=" * 50)

#     # Stop the notebook execution on failure
#     print("\n🛑 STOPPING NOTEBOOK DUE TO JOB FAILURE")
#     sys.exit(1)
# else:
#     print(f"⏰ Job polling stopped with status: {run_status}")
#     final_result = "TIMEOUT"

# print(f"Final status: {run_status}")
# print(f"Total polling time: {(time.time() - start_time)/60:.1f} minutes")
# print(f"Result: {final_result}")
# print("=" * 50)
