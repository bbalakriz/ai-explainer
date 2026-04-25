# Kubernetes object hierarchy for Kubeflow TrainJob (DDP)

## The hierarchy

When you submit a TrainJob, the Training Operator controller creates the following chain of objects:

```
TrainJob  (what you submit)
  └── JobSet  (auto-created by the Training Operator controller)
        └── ReplicatedJob  (a template + a replica count)
              ├── Job 0  (one per node/machine)
              │     └── Pod 0  (one per Job, runs on a physical node)
              │           ├── Process 0  (GPU 0)
              │           └── Process 1  (GPU 1)
              ├── Job 1  (if numNodes were 2)
              │     └── Pod 1
              │           ├── Process 0  (GPU 0)
              │           └── Process 1  (GPU 1)
              ... and so on
```

## What each layer does

### TrainJob

The user facing resource you submit to the cluster. It references a TrainingRuntime and can override values like `numNodes` and the training command.

### JobSet

Auto created by the controller. Always 1 per TrainJob. You never write this yourself. It is the underlying Kubernetes resource that coordinates the full distributed training run.

### ReplicatedJob

Not a running thing. It is a blueprint (a Pod/Job template) plus a replica count. Think of it like a cookie cutter: it says "here is the shape of a Job, and I want N copies of it."

In standard PyTorch DDP, all workers are identical, so you only need one ReplicatedJob (named "node"). You would need multiple ReplicatedJobs only if your training had different roles, for example a parameter server architecture with separate "ps" and "worker" groups.

### Job

One actual running instance stamped out from the ReplicatedJob template. Each Job creates exactly one Pod. The count of Jobs is controlled by `numNodes`. If a Pod crashes, the Job can restart it (via `restartPolicy: OnFailure`).

### Pod

The actual container workload scheduled onto a physical or virtual machine. Each Pod requests GPU, CPU and memory resources and runs the training container plus any init containers.

### Processes

Inside each Pod, `torchrun` spawns multiple worker processes. Each process typically maps to one GPU. The count is controlled by `numProcPerNode`.

## What governs each count

| Layer | Count | Governed by |
|---|---|---|
| JobSet per TrainJob | 1 | always 1:1 (API design) |
| ReplicatedJobs per JobSet | 1 | length of `spec.template.spec.replicatedJobs` array in TrainingRuntime |
| Jobs per ReplicatedJob | N | `numNodes` (TrainJob `spec.trainer.numNodes` overrides TrainingRuntime `spec.mlPolicy.numNodes`) |
| Pods per Job | 1 | always 1 (each Job creates 1 Pod) |
| Processes per Pod | M | `spec.mlPolicy.torch.numProcPerNode` in TrainingRuntime (and matching `torchrun --nproc_per_node` in command) |

Total world size = numNodes x numProcPerNode = N x M

## When you would scale each layer

### More Jobs (increase numNodes)

Scale this when your model or data does not fit on a single machine, or when training on one machine is too slow. For example changing `numNodes` from 1 to 4 gives you 4 machines collaborating via DDP.

### More processes per node (increase numProcPerNode)

Scale this when you add more GPUs to each machine. If a node has 4 GPUs, set `numProcPerNode: 4` and request `nvidia.com/gpu: "4"` in the container resources.

### More ReplicatedJobs

Add a second ReplicatedJob only when your training architecture has distinct roles. Standard DDP does not need this.

## Example: current config in plain english

```
numNodes: 1           --> 1 machine in the training cluster
numProcPerNode: 2     --> 2 training processes on that machine (one per GPU)
nvidia.com/gpu: "2"   --> 2 physical GPUs allocated to that machine's Pod
```

One machine, two GPUs, two processes. Each process handles a slice of each data batch, they synchronize gradients via DDP after each step and the model stays in sync across both.

Scaling to `numNodes: 3` with the same `numProcPerNode: 2` would give 3 machines, 6 GPUs and 6 processes all training one model together.
