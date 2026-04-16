# Serving Models Directly from Hugging Face on OpenShift AI

We will implement deployment of models directly from Hugging Face repositories without needing to download them first. OpenShift AI's vLLM runtime can load models directly from Hugging Face, making deployment faster and eliminating storage requirements.

---

## Prerequisites

Before starting, ensure you have:

1. **OpenShift AI access** with permissions to deploy models
2. **Hugging Face token** (optional but recommended for private models) stored in a Secret
3. **Network access** from your cluster to huggingface.co

---

## Step 1: Prepare Hugging Face Credentials (Optional)

If you're using a private model or want faster downloads, create a secret with your HF token:

```bash
oc create secret generic hf-secret --from-literal=HF_TOKEN=your_hf_token_here
```

---

## Step 2: Navigate to the Project & Start Deployment

1. Log in to the Red Hat OpenShift AI dashboard.
2. Navigate to **Projects** and click on your specific project (e.g., `model-serving`).
3. Scroll down to the **Models** section.
4. Click the **Deploy model** button under Single-model serving to launch the deployment wizard.

---

## Step 3: Model Details

1. **Model location:** Select **URI** from the dropdown.
2. **Model repository:** Enter the Hugging Face model ID (e.g., `hf://Qwen/Qwen2.5-0.5B-Instruct`).
3. **Model type:** Select **Generative AI model (Example, LLM)**.
4. If using a private model, check **Use authentication** and select your `hf-secret`.
5. Click **Next**.

---

## Step 4: Model Deployment (Hardware & Runtime)

1. **Model deployment name:** Enter a descriptive name (e.g., `qwen2.5-0.5b-hf`).
2. **Hardware profile:** Select appropriate gpu-profile.
3. **Serving runtime:** Choose **Select from a list of serving runtimes** and select:
   - **vLLM GPU (x86) ServingRuntime for KServe** (for GPU deployment)
4. **Number of replicas to deploy:** Leave as `1` for testing, increase for production.
5. Click **Next**.

---

## Step 5: Advanced Settings & Deploy

1. Scroll down to the **Configuration parameters** section.
2. Check the box for **Add custom runtime arguments** if needed for specific model configurations:

   For text generation models (default):
   ```text
   --max-model-len=2048
   --tensor-parallel-size=1
   ```

3. **Environment variables** (optional):
   - `HF_TOKEN`: Reference to your secret if not using authentication above
   - `HUGGINGFACE_HUB_CACHE`: `/tmp` (for temporary caching)

4. Click **Deploy** to start the deployment.

---

## Step 6: Monitor Deployment

1. Return to the **Models** section in your project.
2. Watch the deployment status - it may take several minutes for large models to download and initialize.
3. Once deployed, click on the internal and external endpoint link below `Inference endpoints` column of the model list to see the API endpoint URL.

---

## Step 7: Test the Model

Use the provided API endpoint to test your model:

```bash
curl -X POST "https://your-model-endpoint/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-0.5b-hf",
    "prompt": "Hello, how are you?",
    "max_tokens": 100
  }'
```

---

## Supported Model Types

- **Text Generation:** Most common use case (GPT, LLaMA, Qwen, etc.)
- **Encoder Models:** With `--runner=pooling --convert=classify` flags
- **Vision-Language:** Models like LLaVA (ensure runtime supports vision)
- **Custom Architectures:** As long as compatible with vLLM

---

## Advantages of Direct HF Serving

- **No Storage Required:** Models download on-demand
- **Always Latest:** Can update to new model versions easily
- **Faster Deployment:** Skip download step
- **Version Control:** Use specific model commits/tags

---

## Limitations

- **Network Dependent:** Requires stable internet connection
- **Download Time:** Initial deployment slower for large models
- **Rate Limits:** Subject to Hugging Face rate limits
- **Caching:** Models not cached between deployments

---

## Troubleshooting

- **Download fails:** Check network connectivity and HF token
- **Out of memory:** Reduce `max-model-len` or use smaller model
- **Slow inference:** Increase GPU resources or use tensor parallelism
- **Model not found:** Verify the exact model ID on Hugging Face