# Building and Running KFP Pipelines for SFT decoder

We will be implementing, compiling, importing and executing a Kubeflow Pipeline (KFP) for supervised fine tuning (SFT) with S3 based model storage.

---

## Step 1: Update Your Pipeline Script

We will use the `execute_notebook` component in `decoder-kfp-dsl.py` file to accept S3 credentials as secrets. These get injected directly into the operating system environment so the notebook can authenticate natively. Ensure you create a secret named `s3-connection` with the suitable values for the paramaters `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_ENDPOINT_URL`..


## Step 2: Compile the Pipeline

You need to translate the Python pipeline definition into the YAML format that the Kubeflow backend understands.

1. Open your JupyterLab Workbench (your CPU only Control Tower).
2. Open a new Terminal window.
3. Navigate to the folder where you saved `decoder-kfp-dsl.py`.
4. Run the compilation command:

```bash
python decoder-kfp-dsl.py
```

5. You should see the message: `Pipeline compiled successfully to decoder-kfp-dsl-pipeline.yaml!`
6. Right click the generated `decoder-kfp-dsl-pipeline.yaml` file in the Jupyter file browser and select **Download** to save it to your local machine.

---

## Step 3: Import into RHOAI

Now hand the compiled pipeline over to the Data Science Pipeline Server.

1. Go back to your RHOAI Data Science Project dashboard.
2. Scroll down to the **Pipelines** section.
3. Click the **Import pipeline** button.
4. Fill in the details:
   - **Pipeline name**: Enter something descriptive, like `decoder-kfp-dsl-pipeline`.
   - **Pipeline description**: Optional, e.g., "Full SFT pipeline with S3 model upload."
5. Upload the `decoder-kfp-dsl-pipeline.yaml` file you just downloaded.
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
