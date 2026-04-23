# Building and Running KFP Pipelines for SFT Encoder

We will be implementing, compiling, importing and executing a Kubeflow Pipeline (KFP) for supervised fine tuning (SFT) with S3 based model storage.

---

## Step 1: Update Your Pipeline Script

Update the `execute_notebook` component in your `encoder-kfp-dsl.py` file to accept S3 credentials as secrets. These get injected directly into the operating system environment so the notebook can authenticate natively. Ensure you create a secret named `s3-connection` with the suitable values for the paramaters `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_ENDPOINT_URL`..

Add the following parameters and environment variable assignments:

```python
from kfp import dsl
from kfp import compiler
from kfp import kubernetes

...

def execute_notebook(
    repo_url: str, 
    notebook_path: str, 
    mlflow_token: str
) -> str:
    import subprocess
    import os
    import tempfile

    # Only inject non-secret/non-AWS variables here
    os.environ["MLFLOW_TRACKING_TOKEN"] = mlflow_token
    # (AWS variables will be natively injected by Kubernetes before this runs)

    # ... [rest of your git clone and papermill code] ...


# 2. Pipeline definition remains clean
@dsl.pipeline(
    name="llm-sft-eval-merge-pipeline",
    description="Decoder SFT, Evaluation, and Merge Pipeline via Papermill (Secure S3)"
)
def sft_pipeline(
    repo_url: str = "https://github.com/<<org>>/<<repo>>.git", # UPDATE THIS
    mlflow_token: str = "<<token-here>>" # Passed at runtime in the RHOAI UI
):
    
    # ... [define sft_step and eval_step without AWS args] ...

    # Step C: Merge 
    merge_step = execute_notebook(
        repo_url=repo_url,
        notebook_path="encoder-sft/encoder_lora_model_merge.ipynb", 
        mlflow_token=mlflow_token
    ).set_display_name("Merge Weights").after(eval_step)


    # 3. Apply Kubernetes Hardware, Tolerations, PVC Mounts, and SECRETS
    for task in [sft_step, eval_step, merge_step]:
        
        kubernetes.mount_pvc(task, pvc_name="shared-pipeline-data", mount_path="/mnt/data")
        
        # --- ADD THIS SECURE INJECTION BLOCK ---
        kubernetes.use_secret_as_env(
            task,
            secret_name='s3-connection', # Update to your actual OpenShift secret name
            secret_key_to_env={
                'AWS_ACCESS_KEY_ID': 'AWS_ACCESS_KEY_ID',
                'AWS_SECRET_ACCESS_KEY': 'AWS_SECRET_ACCESS_KEY',
                'AWS_ENDPOINT_URL': 'AWS_ENDPOINT_URL' # Maps the K8s secret key to what the notebook expects
            }
        )
        
        task.set_env_variable('S3_BUCKET', 'phayathaibert-model-bucket')
        task.set_env_variable('S3_PREFIX', 'merged-models/from-pipeline')
        # ---------------------------------------
        
        task.set_cpu_request('2').set_cpu_limit('4')
        # ... [rest of your hardware and toleration configs] ...
```

---

## Step 2: Compile the Pipeline

You need to translate the Python pipeline definition into the YAML format that the Kubeflow backend understands.

1. Open your JupyterLab Workbench (your CPU only Control Tower).
2. Open a new Terminal window.
3. Navigate to the folder where you saved `encoder-kfp-dsl.py`.
4. Run the compilation command:

```bash
python encoder-kfp-dsl.py
```

5. You should see the message: `Pipeline compiled successfully to encoder-kfp-dsl-pipeline.yaml!`
6. Right click the generated `encoder-kfp-dsl-pipeline.yaml` file in the Jupyter file browser and select **Download** to save it to your local machine.

---

## Step 3: Import into RHOAI

Now hand the compiled pipeline over to the Data Science Pipeline Server.

1. Go back to your RHOAI Data Science Project dashboard.
2. Scroll down to the **Pipelines** section.
3. Click the **Import pipeline** button.
4. Fill in the details:
   - **Pipeline name**: Enter something descriptive, like `encoder-kfp-dsl-pipeline`.
   - **Pipeline description**: Optional, e.g., "Full SFT pipeline with S3 model upload."
5. Upload the `encoder-kfp-dsl-pipeline.yaml` file you just downloaded.
6. Click **Import**.

---

## Step 4: Execute the Run

This is where everything comes together. You pass your secure credentials at runtime without hardcoding them into Git.

1. From the imported pipeline screen, click **Create run**.
2. **Name**: Give this specific execution a name (e.g., `sft-run-v1`).
3. Scroll down to the **Parameters** section. Because we defined these in the Python script, the UI will automatically display text boxes for each one:

   | Parameter | Value |
   |---|---|
   | `repo_url` | Confirm it points to your Git repository |
   | `mlflow_token` | Paste your actual MLflow tracking token |

4. Click **Create**.

The pipeline server will now spin up the pods, clone the repo, run the SFT step, evaluate it and if successful, the merge notebook will use those newly injected `AWS_` variables to authenticate and push your final model to object storage.
