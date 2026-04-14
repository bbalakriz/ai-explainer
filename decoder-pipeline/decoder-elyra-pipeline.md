# RHOAI Elyra Pipeline: 
## Automated QLoRA Fine-Tuning pipeline for decoder model 

We will implement the **Control Tower** pattern to bypass the `ReadWriteOnce` (RWO) storage deadlock and completely sidesteps Elyra's fragile S3/MinIO artifact zipping mechanics.

---

## Prerequisites

### Checks
- Access to a Red Hat OpenShift AI (RHOAI) cluster with GPU nodes
- A Data Science Project created in RHOAI
- The following notebooks ready in your repository:
  - `decoder_lora_training.ipynb` (SFT fine-tuning)
  - `decoder_lora_evaluation.ipynb` (evaluation)
  - `decoder_lora_model_merge.ipynb` (model merge)

### Configuring the Pipeline Server
Before you touch a workbench or write any code, your project needs its orchestration engine.

1. Navigate to your Project: Open your Data Science Project dashboard in RHOAI.

2. Locate the Pipelines Section: Scroll down to the Pipelines section.

3. Initialize the Server: Click Configure pipeline server.

4. Provide Object Storage: RHOAI requires an object storage connection (Azure blob or OpenShift Data Foundation) to store the pipeline's YAML definitions and lightweight step logs. Enter your credentials:
    - Access Key ID & Secret Access Key
    - Endpoint URL
    - Bucket Name

5. Finalize: Click Configure. Wait a few moments until the server status shows as "Ready." Your project can now accept Elyra submissions.

---

## Phase 0: Repository Setup & Branch Creation

Before anything else, clone the repository and create a dedicated branch for all pipeline-related changes. This keeps your pipeline work isolated and reviewable.

```bash
# Clone the repository
git clone <your-repo-url>
cd <your-repo-name>

# Create and switch to the pipeline branch
git checkout -b pipeline

# Verify you are on the new branch
git branch
```

You should see output like:

```
  main
* pipeline
```

All subsequent changes in this guide will be made on the `pipeline` branch. Once everything is tested and working, you can open a pull request to merge back into `main`.

---

## Phase 1: Notebook Preparation & Version Control

Before touching the Elyra UI, you must prepare your code for headless execution. Elyra pods cannot wait for human input, and they need to know exactly where to find the shared drive.

### Step 1 — Remove Interactive Prompts

Open `decoder_lora_training.ipynb` and `decoder_lora_evaluation.ipynb`. Replace the `getpass()` block with a direct environment variable fetch so the pipeline can authenticate automatically:

```python
import os
os.environ["MLFLOW_TRACKING_TOKEN"] = os.environ.get("MLFLOW_TRACKING_TOKEN", "")
```

### Step 2 — Hardcode Absolute PVC Paths

Update the paths in your notebooks to point directly to the shared drive you are about to create.

- In `decoder_lora_training.ipynb`:
  ```python
  DATA_PATH = "/mnt/data/training_dataset.json"
  OUTPUT_DIR = "/mnt/data/adapters"
  ```
- In `decoder_lora_evaluation.ipynb`:
  ```python
  TRAINING_DATA_PATH = "/mnt/data/training_dataset.json"
  ADAPTER_PATH = "/mnt/data/adapters/latest"
  ```
Move the pip install command for the mlflow package to the topmost cell of the notebook. 

### Step 3 — Commit and Push

Push these updated notebooks (SFT, Evaluation, and Merge) to your Git repository on the `pipeline` branch:

```bash
git add decoder_lora_training.ipynb decoder_lora_evaluation.ipynb
git commit -m "Prepare notebooks for headless Elyra pipeline execution"
git push -u origin pipeline
```

---

## Phase 2: The Control Tower & Data Seeding

We use a lightweight workbench to act as your control tower, ensuring it doesn't hoard expensive GPUs while you author the pipeline.

### Step 1 — Create the Control Tower

In your RHOAI Data Science Project, create a new Workbench using the default profile (or standard Data Science image) with **no GPU** requested.

### Step 2 — Create the Shared Drive

Scroll to the **Cluster storage** section of your project and create a new Persistent Volume Claim (PVC):

| Property | Value |
|----------|-------|
| **Name** | `shared-pipeline-data` |
| **Size** | 20 GB |

### Step 3 — Mount the Drive

Edit your Control Tower workbench and attach the `shared-pipeline-data` PVC, setting the mount path to `/mnt/data`.

### Step 4 — Seed the Data

Start the workbench and open the terminal. Download, copy, or move your training data directly to `/mnt/data/training_dataset.json`.

> **Why we do this:** Bypassing Elyra's default S3 (MinIO) artifact storage completely prevents heavy network transfers, pipeline timeout crashes, and the `NoSuchKey` wildcard errors. The pods will simply read and write to the same physical disk.

### Step 5 — Release the Storage Lock (CRITICAL)

Once the data is copied, **Stop** the workbench. Edit the workbench properties and **Detach** the `shared-pipeline-data` volume.

> **Why we do this:** If your cluster uses `ReadWriteOnce` (RWO) storage, leaving the volume attached to your workbench will permanently lock it. When the pipeline runs, the execution pods will hang forever waiting for the lock to release.

---

## Phase 3: Elyra Pipeline Assembly & Global Configuration

Now we build the pipeline logic and define the global properties that govern the entire execution.

### Step 1 — Start and Clone

Start your Control Tower workbench (now running without the volume attached). Open JupyterLab and clone your Git repository, making sure to check out the `pipeline` branch.

### Step 2 — Assemble the Canvas

Open the Elyra Pipeline Editor. Drag and drop your SFT, Evaluation, and Merge notebooks onto the canvas and draw the connecting arrows to establish the execution order.

### Step 3 — Set Global Node Defaults

Click the **Pipeline Properties** icon (gear symbol) in the Elyra sidebar to set defaults that apply to every node:

| Property | Value |
|----------|-------|
| **Runtime Image** | PyTorch CUDA Python 3.12 |
| **Environment Variables** | `MLFLOW_TRACKING_TOKEN=your_actual_token_here` |
| **Data Volumes — Mount Path** | `/mnt/data` |
| **Data Volumes — PVC Name** | `shared-pipeline-data` |

> The Data Volumes entry dynamically attaches the shared drive to the headless pods at runtime.

---

## Phase 4: Node-Specific Hardware & Execution

Because fine-tuning requires heavy compute, we must explicitly request GPUs and set Kubernetes scheduling rules for each specific step.

### Step 1 — Configure Hardware Resources

Right-click each individual node (SFT, Evals, Merge), select **Properties**, and apply the following resource requests:

| Resource | Request | Limit |
|----------|---------|-------|
| **CPU** | 2 | 4 |
| **Memory** | 4 GB | 8 GB |
| **GPUs** | 1 | — |
| **GPU Vendor** | `nvidia.com/gpu` | — |

### Step 2 — Add Kubernetes Tolerations

In the properties for each node, scroll down to **Kubernetes Tolerations** and add the following rule to schedule the pod on your cluster's GPU nodes:

| Field | Value |
|-------|-------|
| **Key** | `nvidia.com/gpu` |
| **Operator** | `Equal` |
| **Effect** | `NoSchedule` |

### Step 3 — Verify Inheritance

Ensure the following for each node:

- **Runtime Image** is explicitly set to use the pipeline default you defined in Phase 3
- **File Dependencies** is completely empty
- **Output Files** is completely empty

> File Dependencies and Output Files must be empty because the PVC is handling all file transfers — not Elyra's S3/MinIO mechanism.

### Step 4 — Execute

Click the **Run** button, select your Data Science Pipeline Server, and monitor the execution in the RHOAI dashboard.

Your SFT pod will spin up, claim the GPU, read from `/mnt/data`, write the LoRA weights back to `/mnt/data`, and terminate — flawlessly passing the baton to the evaluation and merge steps.

---

## Architecture Summary

```
┌──────────────────────────────────────────────────────┐
├──────────────────────────────────────────────────────┤
│                                                      │
│  Control Tower Workbench (no GPU)                    │
│  ├── Clone repo (pipeline branch)                    │
│  ├── Author Elyra pipeline                           │
│  └── Trigger execution                               │
│                                                      │
│  shared-pipeline-data PVC (20 GB, RWO)               │
│  ├── /mnt/data/training_dataset.json                 │
│  └── /mnt/data/adapters/                             │
│                                                      │
│  Pipeline Execution Pods (GPU-enabled)               │
│  ├── [1] SFT Training    ── reads/writes /mnt/data   │
│  ├── [2] Evaluation       ── reads /mnt/data         │
│  └── [3] Merge            ── reads /mnt/data         │
│                                                      │
└──────────────────────────────────────────────────────┘
```

> **Key insight:** The Control Tower detaches the PVC before pipeline execution begins, allowing each pipeline pod to claim the RWO volume in sequence without deadlocking.
