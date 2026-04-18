from kfp import dsl
from kfp import compiler
from kfp import kubernetes

# 1. Define the reusable Notebook Execution Component
@dsl.component(
    base_image='registry.redhat.io/rhoai/odh-pipeline-runtime-pytorch-cuda-py312-rhel9@sha256:aa457e7394ba73350f5c8e08c56ddba126d9472197e7669c9eb7daa2c02e6777', # Update if your cluster uses a different registry
    packages_to_install=['papermill']
)
def execute_notebook(repo_url: str, notebook_path: str, mlflow_token: str) -> str:
    import subprocess
    import os
    import tempfile

    # Inject the MLflow token for the notebook to pick up
    os.environ["MLFLOW_TRACKING_TOKEN"] = mlflow_token

    # Use tempfile to guarantee a clean, permission-friendly, empty directory every run
    with tempfile.TemporaryDirectory() as workspace_dir:
        print(f"Cloning repository into {workspace_dir}...")
        
        # Clone the repository
        subprocess.run(["git", "clone", repo_url, workspace_dir], check=True)

        # DEBUGGING: Print the entire folder tree to the Kubeflow logs
        # print("\n--- CONTENTS OF CLONED REPOSITORY ---")
        # subprocess.run(["ls", "-laR", workspace_dir])
        # print("-------------------------------------\n")

        target_notebook = os.path.join(workspace_dir, notebook_path)
        
        # Safety check to prevent Papermill from crashing blindly
        if not os.path.exists(target_notebook):
            print(f"CRITICAL ERROR: Could not find {target_notebook}")
            print("Check the 'ls' output above to verify the actual folder structure or branch.")
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
    description="Encoder SFT, Evaluation, and Merge Pipeline via Papermill"
)
def sft_pipeline(
    repo_url: str = "https://github.com/<<org>>/<<repo>>.git", # UPDATE THIS
    mlflow_token: str = "<<token-here>>" # Passed at runtime in the RHOAI UI
):
    # Step A: Training 
    sft_step = execute_notebook(
        repo_url=repo_url,
        notebook_path="encoder-sft/encoder_lora_finetuning.ipynb",
        mlflow_token=mlflow_token
    ).set_display_name("LoRA SFT Training")

    # Step B: Evaluation 
    eval_step = execute_notebook(
        repo_url=repo_url,
        notebook_path="encoder-sft/encoder_lora_evaluation.ipynb",
        mlflow_token=mlflow_token
    ).set_display_name("Model Evaluation").after(sft_step)

    # Step C: Merge 
    merge_step = execute_notebook(
        repo_url=repo_url,
        notebook_path="encoder-sft/encoder_lora_model_merge.ipynb", 
        mlflow_token=mlflow_token
    ).set_display_name("Merge Weights").after(eval_step)


    # 3. Apply Kubernetes Hardware, Tolerations, and PVC Mounts
    for task in [sft_step, eval_step, merge_step]:
        
        # Mount the shared PVC to /mnt/data
        kubernetes.mount_pvc(
            task,
            pvc_name="shared-pipeline-data",
            mount_path="/mnt/data"
        )
        
        # Disable caching for the task
        task.set_caching_options(enable_caching=False)

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
        package_path='encoder-kfp-dsl-pipeline.yaml'
    )
    print("Pipeline compiled successfully to encoder-kfp-dsl-pipeline.yaml!")