# oc apply -f hf-secret.yaml

oc project <<your-project-name>>
oc apply -f pvc.yaml
oc create configmap decoder-fsdp-train-script --from-file=train.py=lora-adapter-builder.py
oc apply -f configmap.yaml
oc apply -f s3-secret.yaml
oc apply -f training-runtime.yaml
oc apply -f trainjob.yaml