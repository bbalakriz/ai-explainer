# Building and Running KFP Pipelines for SFT Encoder

We will be implementing, compiling, importing and executing a Kubeflow Pipeline (KFP) for supervised fine tuning (SFT) with S3 based model storage.

---

## Step 1: Update Your Pipeline Script

Update the `execute_notebook` component in your `encoder-kfp-dsl.py` file to accept AWS credentials as pipeline parameters. These get injected directly into the operating system environment so the notebook can authenticate natively.

Add the following parameters and environment variable assignments:

```python
from kfp import dsl
from kfp import compiler
from kfp import kubernetes

....

def execute_notebook(
    repo_url: str, 
    notebook_path: str, 
    mlflow_token: str,
    aws_access_key: str, # add this
    aws_secret_key: str, # add this
    aws_s3_endpoint: str # add this
) -> str:
    import subprocess
    import os
    import tempfile

    # inject all secrets into the environment for the notebook to pick up natively
    os.environ["MLFLOW_TRACKING_TOKEN"] = mlflow_token
    os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key # add this
    os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_key # add this
    os.environ["AWS_S3_ENDPOINT"] = aws_s3_endpoint # add this
...
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
   | `aws_access_key` | Paste your actual S3 Access Key |
   | `aws_secret_key` | Paste your actual S3 Secret Key |
   | `aws_s3_endpoint` | Provide the URL to your object storage (OpenShift ODF/ Azure) |

4. Click **Create**.

The pipeline server will now spin up the pods, clone the repo, run the SFT step, evaluate it and if successful, the merge notebook will use those newly injected `AWS_` variables to authenticate and push your final model to object storage.
