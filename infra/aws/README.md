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
after six hours on every boot. It creates project-tagged budgets at a $200 operating target and a
$400 emergency threshold. Because AWS billing data is delayed, the launch wrapper also atomically
reserves every worker's worst-case six-hour cost in DynamoDB and refuses a launch that would take
the month above $400. The emergency AWS Budget action stops the active project instances when its
delayed threshold fires. For an intentionally longer job, run
`sudo systemctl disable --now aichessathon-autostop.timer` on that worker and stop it manually.

The reservation gate is deliberately conservative: it does not release reserved dollars when a
worker finishes early. Override the target, ceiling, baseline estimate, or safety window only at
deployment time with `AICHESSATHON_TARGET_BUDGET`, `AICHESSATHON_HARD_BUDGET`,
`AICHESSATHON_BASELINE_SPEND`, and `AICHESSATHON_SAFETY_HOURS`.

## CPU benchmarking

Use a fixed-performance x86 compute-optimized instance; avoid Flex and burstable families for
timing-sensitive comparisons. The scaled default is `c7i.8xlarge` (32 vCPUs). Keep one game worker
per physical core or less for wall-clock calibration:

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

For a sharded run, upload a corpus beneath
`$AICHESSATHON_ARTIFACTS_URI/teacher/runs/RUN_ID/corpus/`, then dispatch one or more Spot CPU
instances with `teacher-worker.sh`. Each process uses one Stockfish thread; completed output shards
are immutable and reruns skip them.

```bash
TEACHER_RUN_ID=pilot-001 TEACHER_TIER=medium TEACHER_NODES=100000 \
TEACHER_SHARDS=8 TEACHER_PARALLELISM=8 bash infra/aws/teacher-worker.sh
```

The scale path launches eight 32-vCPU workers by default, exactly matching the current 256-vCPU
Standard-instance quota:

```bash
bash infra/aws/compute.sh launch-teacher 8
```

Use `TEACHER_LABEL` to annotate the same `deep` input shards at several budgets without collisions,
for example `deep-10k`, `deep-100k`, and `deep-1m`. After all three ladders finish, mine positions
where deeper search changed the value or ordering:

```bash
uv run python -m distill.mine_depth_disagreements \
  --low raw/deep-10k --medium raw/deep-100k --deep raw/deep-1m \
  --select 100000 --output mined
```

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

`distill-gpu-run.sh` downloads a named run's raw records, builds reusable NPZ shards, trains, and
uploads the model under the run-specific S3 prefix:

```bash
DISTILL_RUN_ID=pilot-001 DISTILL_MODEL_NAME=combined DISTILL_EXPECTED_RECORDS=110000 \
  bash infra/aws/distill-gpu-run.sh --epochs 10 --batch-size 1024
```

Reuse a previously built dataset when iterating on the model or loss, avoiding another raw-data
download and conversion pass:

```bash
DISTILL_RUN_ID=pilot-001 DISTILL_MODEL_NAME=fast-top \
DISTILL_REUSE_DATASET_MODEL=medium DISTILL_EXPECTED_RECORDS=100000 \
  bash infra/aws/distill-gpu-run.sh --epochs 15 --batch-size 1024 \
    --accumulator 32 --hidden 32 --bottleneck 16 --top-move-weight 0.25
```

The correctness-gate recipe enforces negamax antisymmetry during training and gives comparisons
involving the teacher's top three moves four times the ranking weight:

```bash
DISTILL_RUN_ID=pilot-001 DISTILL_MODEL_NAME=corrected-64 \
DISTILL_REUSE_DATASET_MODEL=combined DISTILL_EXPECTED_RECORDS=110000 \
  bash infra/aws/distill-gpu-run.sh --epochs 20 --batch-size 1024 \
    --accumulator 64 --hidden 48 --bottleneck 24 \
    --ranking-weight 0.5 --top-move-weight 0.75 --antisymmetry-weight 0.5 \
    --top-k 3 --top-k-ranking-boost 4
```

Compare an exported model and the handcrafted evaluator against retained teacher traces:

```bash
uv run python -m distill.compare_evaluators raw/part-00000.jsonl.gz \
  --model weights/nnue.npz --output evaluator-comparison.json
```

After training, compare the learned evaluator to the identical search using the handcrafted
fallback. Colours are paired for every opening and the report is retained with the run:

```bash
DISTILL_RUN_ID=pilot-001 DISTILL_MODEL_NAME=combined BENCH_ROUNDS=10 \
  bash infra/aws/benchmark-distilled.sh
```

Set `DISTILL_OPPONENT_MODEL_NAME` to compare two learned checkpoints while keeping the search
implementation identical:

```bash
DISTILL_RUN_ID=pilot-001 DISTILL_MODEL_NAME=combined \
DISTILL_OPPONENT_MODEL_NAME=medium BENCH_ROUNDS=10 \
  bash infra/aws/benchmark-distilled.sh
```

## Phase C broad-data and capacity scaling

Phase C deliberately separates research strength from wall-clock deployment. It forms a nested,
feature-disjoint broad-medium dataset with 1M, 3M, and 10M cells, then independently scales the
student to approximately 10, 20, and 40 MB on the 1M component. Every initial cell retains the
Phase B loss, seed, batch size, and 20-epoch recipe. The data axis consumes every shard on every
epoch, so optimizer compute scales with the amount of data; this measures achievable strength
rather than holding optimization cost fixed. The 1M capacity cells still have exactly 500 batches
per epoch.

Create and upload the 9M-position extension on one CPU worker, then label its 2,250 four-thousand
position shards across eight 32-vCPU workers:

```bash
DISTILL_RUN_ID=phase-c-20260902a bash infra/aws/phase-c-corpus.sh

DISTILL_RUN_ID=phase-c-20260902a bash infra/aws/phase-c-label.sh 0
```

Run one worker index from 0 through 7. Workers 1-7 shut down after their assigned labels upload.
Worker 0 waits for all raw shards, builds the exact 2M and 7M components, copies the retained
Phase B 1M component/model into the Phase C namespace, and then shuts down. The preparation step
can also be resumed directly:

```bash
DISTILL_RUN_ID=phase-c-20260902a bash infra/aws/phase-c-prepare.sh
```

Train data-scale cells `D3` and `D10`, or capacity cells `C10`, `C20`, and `C40`:

```bash
DISTILL_RUN_ID=phase-c-20260902a bash infra/aws/phase-c-train.sh D3
```

Pure evaluator scoring remains unconstrained by time. Fixed-node games are the initial search
gate; Phase C does not use wall-clock loss as a reason to stop a promising scaling curve:

```bash
DISTILL_RUN_ID=phase-c-20260902a bash infra/aws/phase-c-evaluate.sh
DISTILL_RUN_ID=phase-c-20260902a bash infra/aws/phase-c-benchmark.sh phase-c-d3m-c4p5
```

## Cost and lifecycle guardrails

1. Set an AWS Budget before launching GPU instances.
2. Use encrypted gp3 volumes and an instance role, never static credentials.
3. Tag instances with project, owner, workload, and expiry.
4. Put datasets/checkpoints in S3; treat EC2 disks as disposable.
5. Stop or terminate instances immediately after jobs. Stopped instances still incur EBS charges.
6. Check regional capacity and current pricing before selecting a family.

Audit both the conservative reservation ledger and delayed AWS Budgets view with:

```bash
bash infra/aws/compute.sh budget-status
```

Stop a reusable worker or terminate a disposable one as soon as the job finishes:

```bash
bash infra/aws/compute.sh stop INSTANCE_ID
bash infra/aws/compute.sh terminate INSTANCE_ID
```

See `requirements.md` before provisioning. `cloudformation.yaml` is the source of truth for shared
resources, and `compute.sh` is the reviewed interface for deployment and instance lifecycle.
