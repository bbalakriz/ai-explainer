# Deploying Fine Tuned Models on OpenShift AI using vLLM

Model Serving transforms your static model weights (stored in S3) into a live API endpoint. In OpenShift AI, this is managed by **KServe**, which relies on two core Kubernetes concepts:

- **ServingRuntime:** The "Engine" template. It defines the software environment (e.g., vLLM) and how models of a certain framework should run.
- **InferenceService:** The "Active Payload." This is your specific deployed model. It links your model files in S3 to a ServingRuntime and defines hardware requirements (Profiles) and runtime arguments.

---

## 1: Deploying the Encoder Model (Phayathaibert)

Encoder models are typically used for classification tasks. Since vLLM defaults to text generation, we must pass specific custom arguments to switch its engine into pooling/classification mode.

### Step 1: Navigate to the Project & Start Deployment

1. Log in to the Red Hat OpenShift AI dashboard.
2. Navigate to **Projects** and click on your specific project (e.g., `encoder-sft`).
3. Scroll down to the **Models** section.
4. Click the **Deploy model** button under Single-model serving to launch the deployment wizard.

### Step 2: Model Details

1. **Model location:** Select **S3 object storage** from the dropdown.
2. Enter your S3 credentials:
   - **Access key / Secret key:** Enter your object storage credentials.
   - **Endpoint:** Provide the URL to your bucket.
   - **Region:** Enter the region (e.g., `us-east-1`).
   - **Bucket:** Enter the bucket name containing your model.
   - **Path:** Enter the folder path pointing to your merged encoder model.
3. Check the box **Create a connection to this location** and give it a Name (e.g., `encoder-s3-conn`).
4. **Model type:** Select **Generative AI model (Example, LLM)**.
5. Click **Next**.

### Step 3: Model Deployment (Hardware & Runtime)

1. **Model deployment name:** Enter `phayathaibert-classifier`.
2. **Hardware profile:** Select **default-profile** (or your standard CPU profile, as encoders can run efficiently on CPUs).
3. **Serving runtime:** Choose **Select from a list of serving runtimes** and select the **vLLM CPU (x86) ServingRuntime for KServe**.
4. **Number of replicas to deploy:** Leave as `1`.
5. Click **Next**.

### Step 4: Advanced Settings & Deploy

1. Scroll down to the **Configuration parameters** section.
2. Check the box for **Add custom runtime arguments**.
3. Paste the following arguments exactly as shown (one per line):

   ```text
   --runner=pooling
   --convert=classify
   --trust-remote-code
   --hf-overrides={"architectures":["RobertaForSequenceClassification"]}
   ```

4. In the **Deployment strategy** section, select **Recreate** instead of **Rolling update**. This ensures that the old pod is terminated before the new one starts, which is safer for model deployments to avoid resource conflicts.
5. Click **Next**, review your settings and click **Deploy model**. Wait for the status to show a green **Loaded** checkmark.

### Step 5: Verify Deployment via Endpoint

Once the model is loaded, you can test the classification endpoint using the provided URL. Run the following `curl` command in your terminal:

```bash
curl https://phayathaibert-encoder-sft.apps.cluster-s99dt.s99dt.sandbox5294.opentlc.com/classify \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phayathaibert",
    "input": "[BOT]: สวัสดีค่ะ ติดต่อเรื่องเปิดใช้งานซิมรายเดือนใช่ไหมคะ [USER]: ใช่ครับ เพิ่งได้รับซิมใหม่มา ยังใช้ไม่ได้เลย เซง สุด แล่ว อ้อแล้วช่วยเช็ก Network แถวบ้านให้หน่อยสัญญานไม่ดีเลย มีปัญหาตลอดโครหงุดหงิด [BOT]: ได้ค่ะ รบกวนแจ้งหมายเลขที่ต้องการเปิดใช้งานก่อนนะคะ [USER]: เบอร์ 086-031-xxxx ครับ [BOT]: ตรวจสอบแล้วเป็นซิมรายเดือนที่รอเปิดใช้งานอยู่ค่ะ ต้องการให้ดำเนินการเปิดใช้งานเลยไหมครับ [USER]: งั้นช่วยเปิดไช้งานเบอร์รายเดือนนี้ให้เลยคับ ต้องการด่วนๆเลยพอดีพรุ่งจะไปต่างจังหวัดพอดีมีประชุมด่วน"
  }' | jq '.data[0] | {label: .label, confidence: (.probs | max)}'
```

Expected output:

```json
{
  "label": "MOBILE_SIM_ACTIVATION_POSTPAID",
  "confidence": 0.953497576713562
}
```

---

## 2: Deploying the Decoder Model (Qwen 3.5)

Decoder models are massive generative engines. Unlike the CPU-bound encoder, Qwen 3.5 requires heavy GPU hardware profiling and strict memory management arguments to prevent Out Of Memory (OOM) crashes.

### Step 1: Navigate to the Project & Start Deployment

1. In your Data Science Project, scroll to the **Models** section.
2. Click the **Deploy model** button again to start a new wizard.

### Step 2: Model Details

1. **Model location:** Select **Existing connection** and choose the S3 connection you created in 1.
2. **Path:** Enter the path to your merged Qwen 3.5 model (e.g., `merged-models/decoder`).
3. **Model type:** Select **Generative AI model (Example, LLM)**.
4. Click **Next**.

### Step 3: Model Deployment (Hardware & Runtime)

1. **Model deployment name:** Enter `qwen25-32b-finetuned`.
2. **Hardware profile:** *CRITICAL CHANGE* -> Select **gpu-profile** (or your cluster's equivalent profile that allocates GPUs).
3. **Serving runtime:** Choose **Select from a list of serving runtimes** and select your **vLLM GPU ServingRuntime**. *(Do not select the CPU version).*
4. **Number of replicas to deploy:** Leave as `1`.
5. Click **Next**.

### Step 4: Advanced Settings & Deploy

1. Check the box for **Add custom runtime arguments**.
2. Paste the following arguments into the text box (one per line):

   ```text
   --max-model-len=2048
   --dtype=half
   --gpu-memory-utilization=0.9
   --enforce-eager
   ```

3. In the **Deployment strategy** section, select **Recreate** instead of **Rolling update**. This ensures that the old pod is terminated before the new one starts, which is safer for GPU-based deployments to avoid resource conflicts.
4. Click **Next**, review the configurations and click **Deploy model**. Wait for the status to show **Loaded**.

#### Understanding the Decoder Arguments

| Argument | Purpose |
|---|---|
| `--max-model-len=2048` | Hard caps the total context window (prompt + response) to 2048 tokens to prevent massive requests from crashing the memory. |
| `--dtype=half` | Forces the model weights to load in FP16 (half precision), cutting the required VRAM in half. |
| `--gpu-memory-utilization=0.9` | Safely reserves 90% of the GPU's VRAM for the model weights and KV Cache, leaving a 10% buffer for PyTorch background operations. |
| `--enforce-eager` | Disables CUDA graph capturing. Eager mode is much more stable for dynamic, variable length API requests. |
| `--tensor-parallel-size=4` | Shards the model across 4 discrete GPUs, distributing the heavy math evenly so the massive model can fit in memory. |

### Step 5: Verify Deployment via Endpoint

For the decoder, you can programmatically fetch the InferenceService (`isvc`) URL using the OpenShift CLI (`oc`) and pass it into your `curl` request:

```bash
# Get the endpoint URL for the decoder
export MODEL_NAME=qwen25-32b-finetuned
export ENDPOINT=$(oc get isvc qwen25-32b-finetuned \
  -n decoder-sft -o jsonpath='{.status.url}')

echo $ENDPOINT
# Example output: https://qwen25-32b-finetuned-decoder-sft.apps.cluster.com

# Test the chat completions API
curl -k -X POST "$ENDPOINT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen25-32b-finetuned",
    "messages": [
      {
        "role": "system",
        "content": "You are an AI intent classifier. Analyze the conversation and output a JSON object with the predicted intent."
      },
      {
        "role": "assistant",
        "content": "สวัสดีค่ะ ติดต่อเรื่องเปิดใช้งานซิมรายเดือนใช่ไหมคะ"
      },
      {
        "role": "user",
        "content": "ใช่ครับ เพิ่งได้รับซิมใหม่มา ยังใช้ไม่ได้เลย เซง สุด แล่ว อ้อแล้วช่วยเช็ก Network แถวบ้านให้หน่อยสัญญานไม่ดีเลย มีปัญหาตลอดโครหงุดหงิด"
      },
      {
        "role": "assistant",
        "content": "ได้ค่ะ รบกวนแจ้งหมายเลขที่ต้องการเปิดใช้งานก่อนนะคะ"
      },
      {
        "role": "user",
        "content": "เบอร์ 086-031-xxxx ครับ"
      },
      {
        "role": "assistant",
        "content": "ตรวจสอบแล้วเป็นซิมรายเดือนที่รอเปิดใช้งานอยู่ค่ะ ต้องการให้ดำเนินการเปิดใช้งานเลยไหมครับ"
      },
      {
        "role": "user",
        "content": "งั้นช่วยเปิดไช้งานเบอร์รายเดือนนี้ให้เลยคับ ต้องการด่วนๆเลยพอดีพรุ่งจะไปต่างจังหวัดพอดีมีประชุมด่วน"
      }
    ],
    "temperature": 0.0,
    "max_tokens": 50,
    "response_format": {
      "type": "json_object"
    }      
  }'
```

Expected output:
```json
{
  "id": "chatcmpl-fb92a1b9...",
  "model": "qwen25-32b-finetuned",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "{\"intent\": \"MOBILE_SIM_ACTIVATION_POSTPAID\"}"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 44,
    "completion_tokens": 19,
    "total_tokens": 63
  }
}
```