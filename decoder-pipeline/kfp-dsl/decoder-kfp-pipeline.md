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

---

## Invoke the compiled pipeline with Kubeflow Pipelines REST APIs

Pipeline runs can be started from the RHOAI Pipelines UI (steps 3 and 4 above) or with the Kubeflow Pipelines REST API on the same pipeline server.

### 1. Resolve the Kubeflow Pipelines API host

Get the OpenShift route hostname for the Data Science Pipelines API in your Data Science project namespace (replace `<your-ds-project>` with the namespace where the pipeline server runs, for example the project that hosts your workbench and DSPA instance). The route name usually follows `ds-pipeline-<dspa-instance-name>`; many clusters use `ds-pipeline-dspa` when the DataSciencePipelinesApplication is named `dspa`. If unsure, run `oc get routes -n <your-ds-project>` and use the route that points at the pipeline API:

```bash
oc get route ds-pipeline-dspa --template='{{ .spec.host }}' -n <your-ds-project>
```

Use the printed host as `<ROUTE>` in the URLs below (for example `ds-pipeline-dspa-<project>.apps.<cluster-domain>`).

### 2. Obtain a Bearer token

The route is protected by OpenShift OAuth. Use a token that can access that namespace:

```bash
oc whoami --show-token
```

Use the output as `<YOUR_TOKEN>` in the `Authorization: Bearer` header.

### 3. Upload the compiled pipeline YAML

Upload `decoder-kfp-dsl-pipeline.yaml` (from step 2) to register a pipeline version. Multipart form fields match the KFP v2beta1 upload contract:

```bash
curl -X POST "https://<ROUTE>/apis/v2beta1/pipelines/upload" \
  -H "Authorization: Bearer <YOUR_TOKEN>" \
  -F "uploadfile=@/path/to/decoder-kfp-dsl-pipeline.yaml" \
  -F "name=decoder-kfp-dsl-pipeline" \
  -F "display_name=Decoder SFT KFP DSL Pipeline" \
  -F "description=Decoder SFT eval and merge pipeline via Papermill with S3" \
  -F "namespace=<your-ds-project>"
```

Field notes:

- `<ROUTE>`: hostname only from step 1; use it inside `https://<ROUTE>/...` as shown.
- `uploadfile`: local path to your compiled YAML.
- `name`: internal id (no spaces).
- `namespace`: same OpenShift project as the pipeline server.

A successful response includes `pipeline_id` (and often a default version id). Keep `pipeline_id` for the run request.

### 4. Start a run with runtime parameters

Trigger execution with a JSON body. Parameters must match the `sft_pipeline` function in `decoder-kfp-dsl.py` (here `repo_url` and `mlflow_token`):

```bash
curl -X POST "https://<ROUTE>/apis/v2beta1/runs" \
  -H "Authorization: Bearer <YOUR_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "pipeline_version_reference": {
      "pipeline_id": "<PIPELINE_ID>"
    },
    "runtime_config": {
      "parameters": {
        "repo_url": "https://github.com/your-org/your-repo.git",
        "mlflow_token": "your-mlflow-tracking-token"
      }
    },
    "display_name": "decoder-sft-run-001"
  }'
```

Replace `<PIPELINE_ID>` with the value from the upload response. Adjust `display_name` and parameter values per run.

If your server expects a specific pipeline version instead of the default, you can set `pipeline_version_reference` to include `pipeline_version_id` as documented in the upstream Kubeflow Pipelines API.

### 5. Verify in the console

In RHOAI, open Pipelines for the same project: you should see the imported pipeline and the new run with the display name you sent.

### Further reading

For listing pipelines, runs, experiments, and other v2beta1 operations, see the [Kubeflow Pipelines API specification (v2beta1)](https://www.kubeflow.org/docs/components/pipelines/reference/api/kubeflow-pipeline-api-spec/). Generic Kubeflow examples often use a `/pipeline/` URL prefix; on RHOAI the `ds-pipeline-dspa` route serves `https://<ROUTE>/apis/v2beta1/...` as in the curl examples above.
