# AWS account prerequisites

Provisioning is intentionally not automated until the team chooses an AWS account, Region, VPC,
budget, and artifact bucket. The execution scripts expect:

- An EC2 instance role with read access to the source/data locations and write access to the chosen
  benchmark/model prefix in S3.
- Systems Manager access for shell sessions, avoiding an inbound SSH rule.
- A budget alert and instance tags identifying the team and workload.
- `AICHESSATHON_ARTIFACTS_URI=s3://bucket/prefix` when results should be uploaded automatically.

Do not put static AWS keys, GitHub tokens, or S3 credentials in this repository or in user data.
