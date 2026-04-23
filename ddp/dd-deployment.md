# DDP LoRA Fine Tuning Deployment Guide

We will be deploying a distributed LoRA fine tuning job using PyTorch DDP (Distributed Data Parallel) on OpenShift AI with the Kubeflow Training Operator.

The pipeline fine tunes the `Qwen/Qwen2.5-7B-Instruct` model using QLoRA (4 bit quantized LoRA) for intent classification. Training data is pulled from S3 at runtime and metrics are tracked in MLflow.

## Files Overview

| File | Purpose |
|------|---------|
| `lora-adapter-builder.py` | The training script. Loads the base model with QLoRA, formats data into chat templates, trains with SFTTrainer and logs to MLflow. |
| `hf-secret.yaml` | Kubernetes Secret for the HuggingFace access token. |
| `s3-secret.yaml` | Kubernetes Secret for S3 credentials (bucket, endpoint, access key, secret key). |
| `pvc.yaml` | Two PersistentVolumeClaims: one for caching the HuggingFace model (20Gi) and one for storing the output LoRA adapters (20Gi). |
| `configmap.yaml` | ConfigMap holding training arguments (model name, output directory). |
| `training-runtime.yaml` | TrainingRuntime that defines the pod template: an init container to download data from S3 and the main GPU training container. |
| `trainjob.yaml` | TrainJob that references the runtime, installs pip dependencies and launches `torchrun` to start training. |
| `deployment.sh` | Convenience script that runs all the `oc` commands in the correct order. |

## Deployment steps

### 1: Select your OpenShift project

Pick or create the project (namespace) where the training job will run. Replace the placeholder with your actual project name.

```bash
oc project <your-project-name>
```

### 2: Create the S3 secret

Open `s3-secret.yaml` and replace each placeholder with the base64 encoded value for your S3 storage.

To encode each value:

```bash
echo -n "your-bucket-name" | base64
echo -n "http://your-s3-endpoint:9000" | base64
echo -n "your-access-key-id" | base64
echo -n "your-secret-access-key" | base64
```

Put each output into the corresponding field:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: s3-secret
type: Opaque
data:
  bucket: <base64-bucket-name>
  endpoint: <base64-endpoint-url>
  aws_access_key_id: <base64-access-key>
  aws_secret_access_key: <base64-secret-key>
```

Then apply it:

```bash
oc apply -f s3-secret.yaml
```

### 3: Create the PersistentVolumeClaims

This creates two 20Gi volumes: one for caching the downloaded HuggingFace model and one for storing the trained LoRA adapters.

```bash
oc apply -f pvc.yaml
```

Verify they are bound:

```bash
oc get pvc hf-model-cache-pvc lora-adapters-pvc
```

### 4: Upload the training script as a ConfigMap

This packages `lora-adapter-builder.py` into a ConfigMap so it gets mounted into the training container at `/app/train.py`.

```bash
oc create configmap decoder-ddp-train-script --from-file=train.py=lora-adapter-builder.py
```

If you need to update the script later, delete and recreate:

```bash
oc delete configmap decoder-ddp-train-script
oc create configmap decoder-ddp-train-script --from-file=train.py=lora-adapter-builder.py
```

### 5: Create the training arguments ConfigMap

This sets the model name and output directory used by the training script. The defaults are `Qwen/Qwen2.5-7B-Instruct` and `/mnt/data/adapters`.

```bash
oc apply -f configmap.yaml
```

### 6: Create the TrainingRuntime

This defines the pod template for training. It includes:

- An init container that installs boto3 and downloads the training dataset from S3 to a shared emptyDir volume
- The main training container using the RHOAI CUDA 12.8 / PyTorch 2.8 image with 2 GPUs, environment variables for HuggingFace caching and volume mounts for the training script, data, adapters and model cache

```bash
oc apply -f training-runtime.yaml
```

### 7: Configure and launch the TrainJob

Before applying, open `trainjob.yaml` and replace these placeholders with your actual values:

1. `<<your-mlflow-token>>`: Your MLflow tracking token (plain text, not base64). You can get this from the OpenShift OAuth token:

```bash
oc whoami -t
```

2. `<<your-project-name>>`: Your OpenShift project name. This appears in two places inside `trainjob.yaml`:
   - The `MLFLOW_WORKSPACE` env var
   - The `MLFLOW_TRACKING_HEADERS` env var (inside the JSON string)

Once the placeholders are filled in:

```bash
oc apply -f trainjob.yaml
```

## S3 Dataset

The `S3_KEY` is set to `shared-data/training_dataset.json` in the training runtime, so make sure the file is uploaded to that path in your S3 bucket.

## Monitoring the Training Job

Check the status of the training job:

```bash
oc get trainjob decoder-ddp-trainjob
```

Watch the pods:

```bash
oc get pods -l trainer.kubeflow.org/trainjob-name=decoder-ddp-trainjob -w
```

View init container logs (data download):

```bash
oc logs <pod-name> -c data-loader
```

View training logs:

```bash
oc logs <pod-name> -c node -f
```

Check GPU utilization across both GPUs while training is running:

```bash
oc exec <pod-name> -c node -- nvidia-smi
```

To watch GPU usage continuously (refreshes every 2 seconds):

```bash
oc exec <pod-name> -c node -- nvidia-smi --loop=2
```

For a compact view showing just utilization and memory per GPU:

```bash
oc exec <pod-name> -c node -- nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
```

With DDP, both GPUs should show similar utilization and memory usage since each GPU runs a full replica of the model.

Training progress and metrics (loss, eval_loss, learning rate) are also available in your MLflow tracking server under the experiment name `decoder-ddp-finetuning`.

## After Training Completes

The trained LoRA adapter is saved to the `lora-adapters-pvc` volume at `/mnt/data/adapters/latest`. A quarantined test split is also saved at `/mnt/data/test_split.jsonl` for offline evaluation.

To retrieve the adapter files, you can launch a temporary pod that mounts the PVC:

```bash
oc run pvc-reader --image=registry.redhat.io/ubi9/ubi-minimal:latest \
  --overrides='{"spec":{"containers":[{"name":"pvc-reader","image":"registry.redhat.io/ubi9/ubi-minimal:latest","command":["sleep","3600"],"volumeMounts":[{"name":"adapters","mountPath":"/mnt/data"}]}],"volumes":[{"name":"adapters","persistentVolumeClaim":{"claimName":"lora-adapters-pvc"}}]}}' \
  --restart=Never

oc exec pvc-reader -- ls /mnt/data/adapters/latest
oc cp pvc-reader:/mnt/data/adapters/latest ./adapter-output
oc delete pod pvc-reader
```

## Using the Deployment Script

If you have already filled in all the placeholders in the YAML files, you can run everything in one shot:

```bash
# edit deployment.sh first to set your project name on the "oc project" line
chmod +x ./deployment.sh
./deployment.sh
```

The script runs these commands in order:

1. Switches to your project
2. Creates the PVCs
3. Packages the training script into a ConfigMap
4. Applies the training arguments ConfigMap
5. Applies the S3 secret
6. Applies the TrainingRuntime
7. Applies the TrainJob (which triggers the training)

## Cleanup

To remove all resources created by this deployment:

```bash
oc delete trainjob decoder-ddp-trainjob
oc delete trainingruntime decoder-ddp-runtime
oc delete configmap decoder-ddp-train-script decoder-ddp-train-args
oc delete secret hf-secret s3-secret
oc delete pvc hf-model-cache-pvc lora-adapters-pvc
```