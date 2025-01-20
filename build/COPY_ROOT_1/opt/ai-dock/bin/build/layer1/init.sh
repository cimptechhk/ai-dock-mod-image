#!/bin/bash

# Must exit and fail to build if any command fails
set -eo pipefail
umask 002

source /opt/ai-dock/bin/build/layer1/common.sh

if [[ "$XPU_TARGET" == "NVIDIA_GPU" ]]; then
    source /opt/ai-dock/bin/build/layer1/nvidia.sh
elif [[ "$XPU_TARGET" == "AMD_GPU" ]]; then
    source /opt/ai-dock/bin/build/layer1/amd.sh
elif [[ "$XPU_TARGET" == "CPU" ]]; then
    source /opt/ai-dock/bin/build/layer1/cpu.sh
else
    printf "No valid XPU_TARGET specified\n" >&2
    exit 1
fi

source /opt/ai-dock/bin/build/layer1/clean.sh