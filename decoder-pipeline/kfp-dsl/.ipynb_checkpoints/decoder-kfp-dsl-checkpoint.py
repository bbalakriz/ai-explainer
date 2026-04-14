from kfp import dsl
from kfp import compiler
from kfp import kubernetes

# 1. Define the reusable Notebook Execution Component
@dsl.component(
    base_image='registry.redhat.io/rhoai/odh-pipeline-runtime-pytorch-cuda-py312-rhel9@sha256:aa457e7394ba73350f5c8e08c56ddba126d9472197e7669c9eb7daa2c02e6777',
    packages_to_install=['papermill']
)
def execute_notebook(
    repo_url: str,
    notebook_path: str,
    mlflow_token: str
) -> str:
    import subprocess
    import os
    import tempfile

    # Inject the MLflow token for the notebook to pick up
    os.environ["MLFLOW_TRACKING_TOKEN"] = mlflow_token
    # Kubernetes natively mounts them as environment variables before this script even runs!

    # Use tempfile to guarantee a clean, permission-friendly, empty directory every run
    with tempfile.TemporaryDirectory() as workspace_dir:
        print(f"Cloning repository into {workspace_dir}...")
        
        # Clone the repository
        subprocess.run(["git", "clone", repo_url, workspace_dir], check=True)

        target_notebook = os.path.join(workspace_dir, notebook_path)
        
        # Safety check to prevent Papermill from crashing blindly
        if not os.path.exists(target_notebook):
            print(f"CRITICAL ERROR: Could not find {target_notebook}")
            return "failed"

        output_nb = f"/mnt/data/executed_{notebook_path.split('/')[-1]}"
        
        # Run the notebook headlessly using Papermill
        try:
            subprocess.run([
                "papermill", 
                target_notebook, 
                output_nb
            ], check=True)
            return "success"
        except subprocess.CalledProcessError as e:
            print(f"Notebook execution failed: {e}")
            return "failed"


# 2. Define the Pipeline
@dsl.pipeline(
    name="llm-sft-eval-merge-pipeline",
    description="Decoder SFT, Evaluation, and Merge Pipeline via Papermill (Secure S3)"
)
def sft_pipeline(
    repo_url: str = "https://github.com/bbalakriz/ai-explainer.git", 
    mlflow_token: str = "sha256~KOlrcQXJ4-J1po8G510HgFsanZh4Tqy15kvrLNoAoN4"
):
    # Step A: Training 
    sft_step = execute_notebook(
        repo_url=repo_url,
        notebook_path="decoder-sft/decoder_lora_finetuning.ipynb",
        mlflow_token=mlflow_token
    ).set_display_name("QLoRA SFT Training")

    # Step B: Evaluation 
    eval_step = execute_notebook(
        repo_url=repo_url,
        notebook_path="decoder-sft/decoder_lora_evaluation.ipynb",
        mlflow_token=mlflow_token
    ).set_display_name("Model Evaluation").after(sft_step)

    # Step C: Merge 
    merge_step = execute_notebook(
        repo_url=repo_url,
        notebook_path="decoder-sft/decoder_lora_model_merge.ipynb", 
        mlflow_token=mlflow_token
    ).set_display_name("Merge Weights").after(eval_step)


    # 3. Apply Kubernetes Hardware, Tolerations, PVC Mounts, and SECRETS
    for task in [sft_step, eval_step, merge_step]:
        
        # Mount the shared PVC to /mnt/data
        kubernetes.mount_pvc(
            task,
            pvc_name="shared-pipeline-data",
            mount_path="/mnt/data"
        )
        
        # ---------------------------------------------------------
        # NEW: Inject AWS S3 credentials from a Kubernetes Secret
        # ---------------------------------------------------------
        kubernetes.use_secret_as_env(
            task,
            secret_name='aws-connection-minio', # <--- UPDATE this to your actual K8s secret name
            secret_key_to_env={
                # 'key_inside_secret': 'NAME_OF_ENV_VAR_IN_POD'
                'AWS_ACCESS_KEY_ID': 'AWS_ACCESS_KEY_ID',
                'AWS_SECRET_ACCESS_KEY': 'AWS_SECRET_ACCESS_KEY',
                'AWS_ENDPOINT_URL': 'AWS_ENDPOINT_URL'
            }
        )
        task.set_env_variable('S3_BUCKET', 'qwen25-3b-model-bucket')
        task.set_env_variable('S3_PREFIX', 'merged-models/decoder')
        
        # Set CPU and Memory constraints
        task.set_cpu_request('2').set_cpu_limit('4')
        task.set_memory_request('4G').set_memory_limit('8G')
        
        # Set GPU allocation
        task.set_accelerator_type('nvidia.com/gpu')
        task.set_gpu_limit('1')
        
        # Add the Toleration for GPU scheduling
        kubernetes.add_toleration(
            task,
            key='nvidia.com/gpu',
            operator='Equal',
            effect='NoSchedule'
        )

# 4. Compile the Pipeline to YAML
if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=sft_pipeline,
        package_path='decoder-kfp-dsl-pipeline.yaml'
    )
    print("Pipeline compiled successfully to decoder-kfp-dsl-pipeline.yaml!")