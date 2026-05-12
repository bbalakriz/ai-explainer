# SDG Pipeline Setup

## Getting a Free Tier API Key (Gemma 4 31B-it)

### 1. Sign in to Google AI Studio

- Go to [Google AI Studio](https://aistudio.google.com) in a web browser.
- Click "Sign in with Google."
- Use a personal Google Account; no separate registration is needed.
- Review and accept the Terms of Service if asked.

### 2. Generate an API Key

- Find the "API key" button on the top left sidebar.
- Click "Create API key" in new project.
- If you have a Google Cloud project, you can select "Create API key in existing project."
- Click the copy icon next to the key after it is created. Keep the key safe because you may not see it again.

### 3. Free Tier Details

- Rate Limits: The free tier for Gemma 4 models allows about 1,500 daily requests.
- Model Identification: Use the model string `gemma-4-31b-it` in the pipeline start command below.
- Regional Availability: Make sure you are signed in from a supported region to access these features.

## Install Dependencies

Before running the pipeline, install the required packages:

```bash
git clone https://github.com/<<repo>>
cd <<repo>>/pipeline

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Pipeline

Once you have your API key from the steps above, export it as an environment variable:

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

Then run the SDG pipeline using the Gemma 4 model:

```bash
cd pipeline

python run.py --model "gemini/gemma-4-31b-it" \
  --api-key $GOOGLE_API_KEY \
  --num-examples 10 --max-seeds 5 --mode flow
```
