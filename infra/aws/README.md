# AWS compute

The recommended split is deliberately simple: EC2 for the first iteration, S3 for datasets and
artifacts, and Systems Manager for access. AWS Batch becomes worthwhile only after jobs need queues,
automatic Spot replacement, or many concurrent machines.

## Local control plane

Install AWS CLI v2 and the Session Manager plugin on the Mac, then authenticate with temporary
browser credentials rather than storing long-lived keys:

```bash
brew install awscli
brew install --cask session-manager-plugin
aws login --profile aichessathon
aws sts get-caller-identity --profile aichessathon
```

This project uses `us-east-1`. Deploy the shared, no-hourly-charge resources once:

```bash
AWS_PROFILE=aichessathon AWS_REGION=us-east-1 bash infra/aws/compute.sh deploy
bash infra/aws/compute.sh outputs
```

The CloudFormation stack creates a private versioned artifact bucket, a narrowly scoped EC2 role,
an outbound-only security group, and CPU/GPU launch templates. It does not launch instances. Both
templates use encrypted gp3 disks, require IMDSv2, have no inbound ports, and schedule a shutdown
after six hours on every boot. For an intentionally longer job, run
`sudo systemctl disable --now aichessathon-autostop.timer` on that worker and stop it manually.

## CPU benchmarking

Use a fixed-performance x86 compute-optimized instance; avoid Flex and burstable families for
timing-sensitive comparisons. The default is `c7i.2xlarge` (8 vCPUs) with three benchmark workers,
leaving headroom for the OS and harness:

```bash
bash infra/aws/compute.sh launch-cpu
bash infra/aws/compute.sh status
bash infra/aws/compute.sh connect INSTANCE_ID
```

On Amazon Linux 2023:

```bash
bash infra/aws/bootstrap-cpu.sh
git clone <team-fork-url> aichessathon
cd aichessathon
BENCH_WORKERS=3 BENCH_ROUNDS=25 bash infra/aws/benchmark.sh
```

Always compare engine versions on the same instance family, AMI, Region, worker count, and time
control. Record the instance type and AMI alongside each retained result. Spot instances are fine
for large statistical runs when interruption is acceptable; use On-Demand for final latency and
clock-margin calibration.

For the offline Stockfish teacher pilot on the same worker:

```bash
sudo bash infra/aws/install-stockfish.sh
TEACHER_NODES=100000 TEACHER_MULTIPV=8 TEACHER_LIMIT=8 \
  bash infra/aws/teacher-pilot.sh
```

The installer verifies the official Stockfish 18 AVX2 archive checksum. Run the pilot before
choosing Batch fleet size; scaling estimates should use measured positions/second, not guesses.

## GPU training

The GPU template uses AWS's current **Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.12 (Amazon
Linux 2023)**. It supplies `/opt/pytorch/bin/python`, the NVIDIA driver, CUDA stack, and SSM agent.
Training framework versions need not equal the competition's CPU runtime; exported ONNX models are
still smoke-tested locally against the pinned competition package versions before submission.

Suggested starting sizes:

- `g6.xlarge`: one NVIDIA L4 with 22 GiB GPU memory; economical for the small value network.
- `g6e.xlarge`: one NVIDIA L40S with 44 GiB GPU memory when larger batches or models justify it.
- P-series instances are unnecessary until profiling proves the workload can use them.

After cloning the exact training commit and downloading the dataset:

```bash
aws s3 cp s3://your-bucket/data/positions.npz training/data/positions.npz
AICHESSATHON_ARTIFACTS_URI=s3://your-bucket/aichessathon \
  bash infra/aws/train.sh --epochs 20 --batch-size 8192
```

Launch only when there is a training job ready. New AWS accounts normally need the adjustable
4-vCPU G-family quota approved first:

```bash
bash infra/aws/compute.sh quota
bash infra/aws/compute.sh launch-gpu       # predictable On-Demand capacity
bash infra/aws/compute.sh launch-gpu-spot  # cheaper, interruption-tolerant jobs
```

`train.sh` makes a small virtual environment that inherits the DLAMI's CUDA-enabled PyTorch rather
than replacing it with the root project's CPU wheel. It verifies CUDA before spending time on the
training job and uploads the ONNX model and metadata when an artifact URI is provided.

The primary sparse evaluator uses `train-distilled.sh` instead:

```bash
DISTILL_DATA=training/data/distilled/100k-nodes \
  bash infra/aws/train-distilled.sh --epochs 10 --batch-size 1024
```

## Cost and lifecycle guardrails

1. Set an AWS Budget before launching GPU instances.
2. Use encrypted gp3 volumes and an instance role, never static credentials.
3. Tag instances with project, owner, workload, and expiry.
4. Put datasets/checkpoints in S3; treat EC2 disks as disposable.
5. Stop or terminate instances immediately after jobs. Stopped instances still incur EBS charges.
6. Check regional capacity and current pricing before selecting a family.

Stop a reusable worker or terminate a disposable one as soon as the job finishes:

```bash
bash infra/aws/compute.sh stop INSTANCE_ID
bash infra/aws/compute.sh terminate INSTANCE_ID
```

See `requirements.md` before provisioning. `cloudformation.yaml` is the source of truth for shared
resources, and `compute.sh` is the reviewed interface for deployment and instance lifecycle.
