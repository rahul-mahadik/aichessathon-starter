# AWS account prerequisites

The checked-in CloudFormation stack provisions the project bucket, EC2 role, security group, and
launch templates in `us-east-1`. The execution scripts expect:

- An EC2 instance role with read access to the source/data locations and write access to the chosen
  benchmark/model prefix in S3.
- Systems Manager access for shell sessions, avoiding an inbound SSH rule.
- A budget alert and instance tags identifying the team and workload.
- `AICHESSATHON_ARTIFACTS_URI=s3://bucket/prefix` when results should be uploaded automatically.
- A 4-vCPU On-Demand or Spot G-family quota before launching the default `g6.xlarge` trainer.

Do not put static AWS keys, GitHub tokens, or S3 credentials in this repository or in user data.
