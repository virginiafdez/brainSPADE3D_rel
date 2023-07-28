#!/usr/bin/bash
#print user info
export USER="${USER:=`whoami`}"
export HOME=/home/$USER

echo "$(id)"

# Define mlflow
export MLFLOW_TRACKING_URI=file:/code/mlruns
echo ${MLFLOW_TRACKING_URI}
export TORCH_HOME=/code/outputs/torch_home

# parse arguments
CMD=""
for i in $@; do
  if [[ $i == *"="* ]]; then
    ARG=${i//=/ }
    CMD=$CMD"--$ARG "
  else
    CMD=$CMD"$i "
  fi
done

# execute command
echo $CMD
$CMD
