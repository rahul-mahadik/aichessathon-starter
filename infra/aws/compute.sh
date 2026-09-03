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

initialize_spend_ledger() {
  local table period baseline item
  table="$(stack_output SpendReservationsTableName)"
  period="$(date -u +%Y-%m)"
  baseline="$(stack_output BaselineProjectSpend)"
  item="{\"BillingPeriod\":{\"S\":\"$period\"},\"ReservedUSD\":{\"N\":\"$baseline\"},\"BaselineUSD\":{\"N\":\"$baseline\"}}"
  if aws_project dynamodb put-item \
    --table-name "$table" \
    --item "$item" \
    --condition-expression 'attribute_not_exists(BillingPeriod)' >/dev/null 2>&1; then
    echo "Initialized $period project spend reservation at \$$baseline."
  fi
}

reserve_launch_budget() {
  local workload="$1" count="$2" rate_key rate hours amount hard target remaining period table now values total
  case "$workload" in
    cpu|teacher) rate_key=CpuHourlyRateCeiling ;;
    gpu) rate_key=GpuHourlyRateCeiling ;;
    *) echo "unsupported reservation workload: $workload" >&2; exit 2 ;;
  esac
  rate="$(stack_output "$rate_key")"
  hours="$(stack_output WorkerSafetyHours)"
  hard="$(stack_output HardBudgetLimit)"
  target="$(stack_output TargetBudgetLimit)"
  amount="$(awk -v rate="$rate" -v hours="$hours" -v count="$count" 'BEGIN {printf "%.2f", rate * hours * count}')"
  remaining="$(awk -v hard="$hard" -v amount="$amount" 'BEGIN {printf "%.2f", hard - amount}')"
  period="$(date -u +%Y-%m)"
  table="$(stack_output SpendReservationsTableName)"
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  values="{\":amount\":{\"N\":\"$amount\"},\":remaining\":{\"N\":\"$remaining\"},\":now\":{\"S\":\"$now\"},\":workload\":{\"S\":\"$workload\"}}"
  if ! total="$(aws_project dynamodb update-item \
    --table-name "$table" \
    --key "{\"BillingPeriod\":{\"S\":\"$period\"}}" \
    --update-expression 'SET LastReservationAt = :now, LastWorkload = :workload ADD ReservedUSD :amount' \
    --condition-expression 'attribute_not_exists(ReservedUSD) OR ReservedUSD <= :remaining' \
    --expression-attribute-values "$values" \
    --return-values ALL_NEW \
    --query 'Attributes.ReservedUSD.N' --output text)"; then
    local current
    current="$(aws_project dynamodb get-item \
      --table-name "$table" \
      --key "{\"BillingPeriod\":{\"S\":\"$period\"}}" \
      --query 'Item.ReservedUSD.N' --output text)"
    echo "Refusing launch: reserving \$$amount would exceed the \$$hard project ceiling (currently \$$current reserved)." >&2
    exit 1
  fi
  RESERVATION_AMOUNT="$amount"
  echo "Reserved worst-case launch cost: \$$amount; month total: \$$total / \$$hard."
  if awk -v total="$total" -v target="$target" 'BEGIN {exit !(total > target)}'; then
    echo "Notice: authorized reservations are now above the \$$target operating target." >&2
  fi
}

release_launch_budget() {
  local amount="$1" period table values
  period="$(date -u +%Y-%m)"
  table="$(stack_output SpendReservationsTableName)"
  values="{\":amount\":{\"N\":\"-$amount\"}}"
  aws_project dynamodb update-item \
    --table-name "$table" \
    --key "{\"BillingPeriod\":{\"S\":\"$period\"}}" \
    --update-expression 'ADD ReservedUSD :amount' \
    --expression-attribute-values "$values" >/dev/null
}

register_hard_stop_action() {
  local account_id budget_name role_arn topic_arn ids definition action_id
  account_id="$(aws_project sts get-caller-identity --query Account --output text)"
  budget_name="$(stack_output HardBudgetName)"
  role_arn="$(stack_output BudgetActionRoleArn)"
  topic_arn="$(stack_output BudgetAlertsTopicArn)"
  ids="$(aws_project ec2 describe-instances \
    --filters "Name=tag:Project,Values=$PROJECT_NAME" \
      Name=instance-state-name,Values=pending,running \
    --query 'Reservations[].Instances[].InstanceId' --output json)"
  if [[ "$ids" == "[]" ]]; then
    return
  fi
  definition="$(jq -nc --arg region "$AWS_REGION_NAME" --argjson ids "$ids" \
    '{SsmActionDefinition:{ActionSubType:"STOP_EC2_INSTANCES",Region:$region,InstanceIds:$ids}}')"
  action_id="$(aws_project budgets describe-budget-actions-for-budget \
    --account-id "$account_id" --budget-name "$budget_name" \
    --query 'Actions[0].ActionId' --output text)"
  if [[ "$action_id" == "None" ]]; then
    aws_project budgets create-budget-action \
      --account-id "$account_id" \
      --budget-name "$budget_name" \
      --notification-type ACTUAL \
      --action-type RUN_SSM_DOCUMENTS \
      --action-threshold ActionThresholdValue=100,ActionThresholdType=PERCENTAGE \
      --definition "$definition" \
      --execution-role-arn "$role_arn" \
      --approval-model AUTOMATIC \
      --subscribers "SubscriptionType=SNS,Address=$topic_arn" >/dev/null
    echo "Registered automatic \$$(stack_output HardBudgetLimit) EC2 stop action."
  else
    aws_project budgets update-budget-action \
      --account-id "$account_id" \
      --budget-name "$budget_name" \
      --action-id "$action_id" \
      --notification-type ACTUAL \
      --action-threshold ActionThresholdValue=100,ActionThresholdType=PERCENTAGE \
      --definition "$definition" \
      --execution-role-arn "$role_arn" \
      --approval-model AUTOMATIC \
      --subscribers "SubscriptionType=SNS,Address=$topic_arn" >/dev/null
    echo "Updated automatic emergency stop action for active project instances."
  fi
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
  local vpc_id subnet_id target_budget hard_budget
  target_budget="${AICHESSATHON_TARGET_BUDGET:-500}"
  hard_budget="${AICHESSATHON_HARD_BUDGET:-1000}"
  if awk -v target="$target_budget" -v hard="$hard_budget" 'BEGIN {exit !(hard <= target)}'; then
    echo "Hard budget must be greater than the operating target." >&2
    exit 2
  fi
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
      CpuInstanceType="${AICHESSATHON_CPU_TYPE:-c7i.8xlarge}" \
      GpuInstanceType="${AICHESSATHON_GPU_TYPE:-g6.xlarge}" \
      TargetBudgetLimit="$target_budget" \
      HardBudgetLimit="$hard_budget" \
      BaselineProjectSpend="${AICHESSATHON_BASELINE_SPEND:-20}" \
      CpuHourlyRateCeiling="${AICHESSATHON_CPU_RATE_CEILING:-2.25}" \
      GpuHourlyRateCeiling="${AICHESSATHON_GPU_RATE_CEILING:-2.50}" \
      WorkerSafetyHours="${AICHESSATHON_SAFETY_HOURS:-6}" \
    --tags Project="$PROJECT_NAME" ManagedBy=cloudformation

  if ! aws_project ce update-cost-allocation-tags-status \
    --cost-allocation-tags-status TagKey=Project,Status=Active >/dev/null; then
    echo "Warning: could not activate Project as a cost-allocation tag; AWS Budget data may remain account-wide until activated." >&2
  fi
  initialize_spend_ledger
  echo "Artifact URI: $(stack_output ArtifactUri)"
}

launch() {
  local workload="$1" market="${2:-on-demand}" count="${3:-1}" output_key instance_ids template_id tag_workload
  local subnet_id="${AICHESSATHON_SUBNET_ID:-}"
  local instance_type="${AICHESSATHON_INSTANCE_TYPE:-}"
  if [[ ! "$count" =~ ^[1-9][0-9]*$ ]] || (( count > 100 )); then
    echo "Count must be an integer from 1 to 100." >&2
    exit 2
  fi
  case "$workload" in
    cpu) output_key=CpuLaunchTemplateId; tag_workload=benchmark ;;
    teacher) output_key=CpuLaunchTemplateId; tag_workload=teacher ;;
    gpu) output_key=GpuLaunchTemplateId ;;
    *) echo "Workload must be cpu, teacher, or gpu" >&2; exit 2 ;;
  esac
  tag_workload="${tag_workload:-training}"
  template_id="$(stack_output "$output_key")"
  if [[ -n "$subnet_id" ]]; then
    if [[ ! "$subnet_id" =~ ^subnet-[0-9a-f]+$ ]]; then
      echo "AICHESSATHON_SUBNET_ID is not a valid subnet ID." >&2
      exit 2
    fi
    aws_project ec2 describe-subnets --subnet-ids "$subnet_id" >/dev/null
  fi
  if [[ -n "$instance_type" && ! "$instance_type" =~ ^[a-z0-9][a-z0-9.]+$ ]]; then
    echo "AICHESSATHON_INSTANCE_TYPE is not a valid EC2 instance type." >&2
    exit 2
  fi
  run_instances() {
    if [[ -n "$subnet_id" && -n "$instance_type" ]]; then
      aws_project ec2 run-instances "$@" --subnet-id "$subnet_id" --instance-type "$instance_type"
    elif [[ -n "$subnet_id" ]]; then
      aws_project ec2 run-instances "$@" --subnet-id "$subnet_id"
    elif [[ -n "$instance_type" ]]; then
      aws_project ec2 run-instances "$@" --instance-type "$instance_type"
    else
      aws_project ec2 run-instances "$@"
    fi
  }
  reserve_launch_budget "$workload" "$count"

  if [[ "$market" == "spot" ]]; then
    if ! instance_ids="$(run_instances \
      --launch-template "LaunchTemplateId=$template_id,Version=\$Latest" \
      --instance-initiated-shutdown-behavior terminate \
      --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}' \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Workload,Value=$tag_workload}]" \
      --count "$count" --query 'Instances[].InstanceId' --output text)"; then
      release_launch_budget "$RESERVATION_AMOUNT"
      exit 1
    fi
  elif [[ "$market" == "on-demand" ]]; then
    local shutdown_behavior=stop
    if [[ "$workload" == "teacher" ]]; then
      shutdown_behavior=terminate
    fi
    if ! instance_ids="$(run_instances \
      --launch-template "LaunchTemplateId=$template_id,Version=\$Latest" \
      --instance-initiated-shutdown-behavior "$shutdown_behavior" \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Workload,Value=$tag_workload}]" \
      --count "$count" --query 'Instances[].InstanceId' --output text)"; then
      release_launch_budget "$RESERVATION_AMOUNT"
      exit 1
    fi
  else
    release_launch_budget "$RESERVATION_AMOUNT"
    echo "Market must be on-demand or spot" >&2
    exit 2
  fi

  read -r -a launched_ids <<<"$instance_ids"
  aws_project ec2 wait instance-running --instance-ids "${launched_ids[@]}"
  register_hard_stop_action
  echo "Launched $workload instances ($market): $instance_ids"
  if [[ "$market" == "spot" ]]; then
    echo "They will terminate on shutdown or after the safety window."
  elif [[ "$workload" == "teacher" ]]; then
    echo "They will terminate on shutdown or after the safety window."
  else
    echo "They will stop automatically after the safety window."
  fi
  echo "Check readiness with: bash infra/aws/compute.sh status"
  echo "Connect with: bash infra/aws/compute.sh connect ${launched_ids[0]}"
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

gpu_quota() {
  aws_project service-quotas list-service-quotas \
    --service-code ec2 \
    --query 'Quotas[?QuotaCode==`L-DB2E81BA` || QuotaCode==`L-3819A6DF`].{Name:QuotaName,vCPUs:Value}' \
    --output table
}

budget_status() {
  local account_id period table
  account_id="$(aws_project sts get-caller-identity --query Account --output text)"
  period="$(date -u +%Y-%m)"
  table="$(stack_output SpendReservationsTableName)"
  echo "Worst-case project launch reservations:"
  aws_project dynamodb get-item \
    --table-name "$table" \
    --key "{\"BillingPeriod\":{\"S\":\"$period\"}}" \
    --query 'Item.{Period:BillingPeriod.S,ReservedUSD:ReservedUSD.N,BaselineUSD:BaselineUSD.N,LastReservationAt:LastReservationAt.S,LastWorkload:LastWorkload.S}' \
    --output table
  echo "Project-tagged AWS budgets (billing data is delayed):"
  aws_project budgets describe-budget \
    --account-id "$account_id" --budget-name "$(stack_output TargetBudgetName)" \
    --query 'Budget.{Name:BudgetName,Limit:BudgetLimit.Amount,Actual:CalculatedSpend.ActualSpend.Amount,Forecast:CalculatedSpend.ForecastedSpend.Amount}' \
    --output table
  aws_project budgets describe-budget \
    --account-id "$account_id" --budget-name "$(stack_output HardBudgetName)" \
    --query 'Budget.{Name:BudgetName,Limit:BudgetLimit.Amount,Actual:CalculatedSpend.ActualSpend.Amount,Forecast:CalculatedSpend.ForecastedSpend.Amount}' \
    --output table
}

case "${1:-help}" in
  deploy) deploy ;;
  launch-cpu) launch cpu on-demand "${2:-1}" ;;
  launch-cpu-spot) launch cpu spot "${2:-1}" ;;
  launch-teacher) launch teacher on-demand "${2:-8}" ;;
  launch-teacher-spot) launch teacher spot "${2:-8}" ;;
  launch-gpu) launch gpu on-demand "${2:-1}" ;;
  launch-gpu-spot) launch gpu spot "${2:-1}" ;;
  status) status ;;
  budget-status) budget_status ;;
  register-budget-action) register_hard_stop_action ;;
  quota) gpu_quota ;;
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
    echo "Commands: deploy, launch-cpu [COUNT], launch-cpu-spot [COUNT],"
    echo "          launch-teacher [COUNT], launch-teacher-spot [COUNT],"
    echo "          launch-gpu [COUNT], launch-gpu-spot [COUNT], status, budget-status,"
    echo "          register-budget-action,"
    echo "          quota, outputs, connect INSTANCE, start INSTANCE, stop INSTANCE, terminate INSTANCE"
    ;;
esac
