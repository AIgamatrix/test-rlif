set +x

source ./ray_utils.sh

# ===== HARDCODED CLUSTER CONFIGURATION =====
# Modify these lines for your specific cluster setup
HEAD_NODE=NODE9
WORKER_NODES=(
    NODE10
)


# ===== RAY TEMPORARY DIRECTORY CONFIGURATION =====
# Specify the custom Ray temporary directory path here
# This is useful when /tmp has limited space or you want to use a faster/larger filesystem
export RAY_TMPDIR="/opt/dlami/nvme/ray_tmp"  # Change this to your desired path

# ===== ENVIRONMENT CONFIGURATION =====
export ACTIVATION_PATH="/path/to/Intuitor/verl-intuitor/env_verl/bin/activate"
export RAY_SSH_USER="ubuntu"
GPUS_PER_NODE=8
# ==========================================

# Set up the Ray cluster configuration
set_ray_cluster "$HEAD_NODE" "${WORKER_NODES[@]}"

# Calculate total nodes
TOTAL_NODES=$((1 + ${#WORKER_NODES[@]}))

echo "Training configuration:"
echo "  Total nodes: $TOTAL_NODES"
echo "  GPUs per node: $GPUS_PER_NODE"
echo "  Total GPUs: $((TOTAL_NODES * GPUS_PER_NODE))"
echo "  Ray temp directory: $RAY_TMPDIR"

# Check if cluster is already running
if check_ray_cluster &>/dev/null; then
    echo "Ray cluster is already running. Using existing cluster."
else
    echo "Starting Ray cluster..."
    if ! start_ray_cluster; then
        echo "Failed to start Ray cluster. Exiting."
        exit 1
    fi
fi

# Get dashboard address
DASHBOARD_ADDRESS="http://${HEAD_NODE}:${RAY_DASHBOARD_PORT}"
echo "Ray dashboard address: $DASHBOARD_ADDRESS"