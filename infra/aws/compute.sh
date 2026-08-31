#!/usr/bin/env bash
set -euo pipefail

AWS_PROFILE_NAME="${AWS_PROFILE:-aichessathon}"
AWS_REGION_NAME="${AWS_REGION:-us-east-1}"
STACK_NAME="${AICHESSATHON_STACK:-aichessathon-compute}"
PROJECT_NAME="${AICHESSATHON_PROJECT:-aichessathon}"
GIT_REPOSITORY_URL="${AICHESSATHON_GIT_URL:-https://github.com/rahul-mahadik/aichessathon-starter.git}"
GIT_REF_NAME="${AICHESSATHON_GIT_REF:-codex/environment-setup}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

aws_project() {
  aws --profile "$AWS_PROFILE_NAME" --region "$AWS_REGION_NAME" "$@"
}

stack_output() {
  local key="$1"
  aws_project cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --query "Stacks[0].Outputs[?OutputKey=='$key'].OutputValue | [0]" \
    --output text
}

require_managed_instance() {
  local instance_id="$1"
  local project_tag
  project_tag="$(aws_project ec2 describe-tags \
    --filters "Name=resource-id,Values=$instance_id" "Name=key,Values=Project" \
    --query 'Tags[0].Value' --output text)"
  if [[ "$project_tag" != "$PROJECT_NAME" ]]; then
    echo "Refusing: $instance_id is not tagged Project=$PROJECT_NAME" >&2
    exit 1
  fi
}

deploy() {
  local vpc_id subnet_id
  vpc_id="$(aws_project ec2 describe-vpcs \
    --filters Name=is-default,Values=true --query 'Vpcs[0].VpcId' --output text)"
  subnet_id="$(aws_project ec2 describe-subnets \
    --filters Name=vpc-id,Values="$vpc_id" Name=availability-zone,Values="${AWS_REGION_NAME}a" \
    --query 'Subnets[0].SubnetId' --output text)"
  if [[ "$vpc_id" == "None" || "$subnet_id" == "None" ]]; then
    echo "A default VPC and a subnet in ${AWS_REGION_NAME}a are required." >&2
    exit 1
  fi

  aws_project cloudformation deploy \
    --stack-name "$STACK_NAME" \
    --template-file "$SCRIPT_DIR/cloudformation.yaml" \
    --capabilities CAPABILITY_IAM \
    --parameter-overrides \
      ProjectName="$PROJECT_NAME" \
      VpcId="$vpc_id" \
      SubnetId="$subnet_id" \
      GitRepositoryUrl="$GIT_REPOSITORY_URL" \
      GitRef="$GIT_REF_NAME" \
    --tags Project="$PROJECT_NAME" ManagedBy=cloudformation

  echo "Artifact URI: $(stack_output ArtifactUri)"
}

launch() {
  local workload="$1" market="${2:-on-demand}" output_key instance_id template_id
  case "$workload" in
    cpu) output_key=CpuLaunchTemplateId ;;
    gpu) output_key=GpuLaunchTemplateId ;;
    *) echo "Workload must be cpu or gpu" >&2; exit 2 ;;
  esac
  template_id="$(stack_output "$output_key")"

  if [[ "$market" == "spot" ]]; then
    instance_id="$(aws_project ec2 run-instances \
      --launch-template "LaunchTemplateId=$template_id,Version=\$Latest" \
      --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}' \
      --count 1 --query 'Instances[0].InstanceId' --output text)"
  elif [[ "$market" == "on-demand" ]]; then
    instance_id="$(aws_project ec2 run-instances \
      --launch-template "LaunchTemplateId=$template_id,Version=\$Latest" \
      --count 1 --query 'Instances[0].InstanceId' --output text)"
  else
    echo "Market must be on-demand or spot" >&2
    exit 2
  fi

  echo "Launched $workload instance: $instance_id ($market)"
  echo "It will stop automatically after six hours unless you cancel/reschedule shutdown."
  echo "Check readiness with: bash infra/aws/compute.sh status"
  echo "Connect with: bash infra/aws/compute.sh connect $instance_id"
}

status() {
  aws_project ec2 describe-instances \
    --filters \
      "Name=tag:Project,Values=$PROJECT_NAME" \
      Name=tag:ManagedBy,Values=cloudformation \
      Name=instance-state-name,Values=pending,running,stopping,stopped \
    --query 'Reservations[].Instances[].{Name:Tags[?Key==`Name`]|[0].Value,Id:InstanceId,State:State.Name,Type:InstanceType,Launched:LaunchTime}' \
    --output table
}

case "${1:-help}" in
  deploy) deploy ;;
  launch-cpu) launch cpu on-demand ;;
  launch-cpu-spot) launch cpu spot ;;
  launch-gpu) launch gpu on-demand ;;
  launch-gpu-spot) launch gpu spot ;;
  status) status ;;
  connect)
    require_managed_instance "${2:?instance id required}"
    aws_project ssm start-session --target "$2"
    ;;
  start)
    require_managed_instance "${2:?instance id required}"
    aws_project ec2 start-instances --instance-ids "$2"
    ;;
  stop)
    require_managed_instance "${2:?instance id required}"
    aws_project ec2 stop-instances --instance-ids "$2"
    ;;
  terminate)
    require_managed_instance "${2:?instance id required}"
    aws_project ec2 terminate-instances --instance-ids "$2"
    ;;
  outputs)
    aws_project cloudformation describe-stacks --stack-name "$STACK_NAME" \
      --query 'Stacks[0].Outputs[].{Key:OutputKey,Value:OutputValue}' --output table
    ;;
  help|*)
    echo "Usage: bash infra/aws/compute.sh <command>"
    echo "Commands: deploy, launch-cpu, launch-cpu-spot, launch-gpu, launch-gpu-spot,"
    echo "          status, outputs, connect INSTANCE, start INSTANCE, stop INSTANCE, terminate INSTANCE"
    ;;
esac
