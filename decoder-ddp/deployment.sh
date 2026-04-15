
oc project <<your-project-name>>

################################################################################################################
# uncomment the following lines to create PVCs and secret for Hugging Face model cache and LoRA adapters if needed
# oc create pvc hf-model-cache-pvc --claim-name=hf-model-cache-pvc --size=20Gi
# oc create pvc lora-adapters-pvc --claim-name=lora-adapters-pvc --size=20Gi
# oc apply -f hf-secret.yaml
################################################################################################################

oc create configmap decoder-ddp-train-script --from-file=train.py=lora-adapter-builder.py
oc apply -f configmap.yaml
oc apply -f s3-secret.yaml
oc apply -f training-runtime.yaml
oc apply -f trainjob.yaml