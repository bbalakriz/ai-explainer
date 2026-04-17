# Downloading Models to PVC for OpenShift AI Serving

We will implement downloading a pre-trained model from Hugging Face directly into a Persistent Volume Claim (PVC) in your OpenShift cluster. This approach eliminates the need for external storage like S3 and allows you to serve models directly from cluster storage.

---

## Prerequisites

Before starting, ensure you have:

1. **OpenShift AI access** with permissions to create Jobs and PVCs
2. **Hugging Face token** stored in a Secret named `hf-secret` with key `HF_TOKEN`
3. **PVC created** named `models-storage-pvc` with sufficient storage (e.g., 10Gi)

---

## Step 1: Create the PVC (if not exists)

If you haven't created the PVC yet, apply this YAML:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: models-storage-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
  storageClassName: ocs-storagecluster-ceph-rbd  # Adjust based on your cluster
```

Apply it with: `oc apply -f pvc.yaml`

---

## Step 2: Create the Hugging Face Secret

Create a secret with your Hugging Face token:

```bash
oc create secret generic hf-secret --from-literal=HF_TOKEN=your_hf_token_here
```

---

## Step 3: Download the Model

Use the provided `model-downloader.yaml` Job to download a model. This example downloads Qwen2.5-0.5B-Instruct:

```yaml
kind: Job
apiVersion: batch/v1
metadata:
  name: model-downloader
spec:
  manualSelector: false
  backoffLimit: 2
  completions: 1
  template:
    spec:
      restartPolicy: OnFailure
      volumes:
        - name: storage
          persistentVolumeClaim:
            claimName: models-storage-pvc
      containers:
        - name: downloader
          image: 'registry.redhat.io/ubi9/python-312:latest'
          command:
            - bash
            - '-c'
            - |
              pip install --no-cache-dir huggingface_hub[hf_transfer]

              python3 - <<'PY'
              import os
              from huggingface_hub import snapshot_download

              # Configuration - MODIFY THESE VALUES
              repo_id = "Qwen/Qwen2.5-0.5B-Instruct"  # Change to your desired model
              local_dir = "/mnt/models/qwen2.5-0.5b"  # Change to your desired path

              # Enable fast transfer
              os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

              print(f"Starting download of {repo_id} to {local_dir}...")

              try:
                  snapshot_download(
                      repo_id=repo_id,
                      local_dir=local_dir,
                      local_dir_use_symlinks=False,
                      repo_type="model"
                  )
                  print("Download complete.")
              except Exception as e:
                  print(f"Error during download: {e}")
                  exit(1)
              PY

              echo "Setting permissions..."
              chmod -R 777 /mnt/models/qwen2.5-0.5b  # Adjust path

          env:
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-secret
                  key: HF_TOKEN
          resources:
            limits:
              cpu: '4'
              memory: 8Gi
            requests:
              cpu: '2'
              memory: 4Gi
          volumeMounts:
            - name: storage
              mountPath: /mnt/models
```

Apply the job: `oc apply -f model-downloader.yaml`

Monitor the job: `oc get jobs` and `oc logs job/model-downloader`

---

## Step 4: Verify the Download

Once the job completes successfully, verify the model files are in the PVC:

```bash
# spin up a temporary pod that mounts the PVC and list the contents
oc run debug-pod --image=registry.redhat.io/ubi9/ubi-minimal --rm -it \
  --overrides='{
    "spec": {
      "containers": [{
        "name": "debug-pod",
        "image": "registry.redhat.io/ubi9/ubi-minimal",
        "command": ["bash"],
        "stdin": true,
        "tty": true,
        "volumeMounts": [{
          "name": "model-vol",
          "mountPath": "/mnt/models"
        }]
      }],
      "volumes": [{
        "name": "model-vol",
        "persistentVolumeClaim": {
          "claimName": "models-storage-pvc"
        }
      }]
    }
  }' -- bash

# inside the pod
ls -la /mnt/models/
```

You should see your model directory with the downloaded files.

---

## Step 5: Get the PVC URI for Model Serving

For OpenShift AI model serving using PVC, the URI format is:

```
pvc://models-storage-pvc/path/to/model
```

For our example: `pvc://models-storage-pvc/qwen2.5-0.5b`

---

## Step 6: Deploy the Model Using PVC

In the OpenShift AI dashboard:

1. Go to your project → Models → Deploy model
2. **Model location:** Select **URI** from the dropdown
3. **URI:** Set the model path `pvc://models-storage-pvc/qwen2.5-0.5b`
4. **Model type:** Select appropriate type (e.g., Generative AI model)
5. Continue with deployment settings as usual

The model will now be served directly from your cluster's PVC storage.

