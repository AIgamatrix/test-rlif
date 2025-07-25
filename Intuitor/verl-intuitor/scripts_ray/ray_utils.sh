#!/bin/bash
# ray_utils.sh - Ray cluster management utilities with activation script support

# Configuration
RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_SSH_USER="${RAY_SSH_USER:-ubuntu}"
ACTIVATION_PATH="${ACTIVATION_PATH:-/fsx/ubuntu/miniconda3/bin/activate}"  # Path to activation script
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray}"  # Ray temporary directory

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Global variables (will be set by set_ray_cluster)
RAY_HEAD_NODE=""
RAY_WORKER_NODES=()
ALL_NODES=()
TOTAL_NODES=0

# Function to create activation command with RAY_TMPDIR
get_activation_command() {
    echo "export RAY_TMPDIR='${RAY_TMPDIR}' && source ${ACTIVATION_PATH}"
}


CONDA_ENV="${CONDA_ENV:-base}"  # Set your conda environment name
CONDA_PATH="${CONDA_PATH:-/fsx/ubuntu/miniconda3}"  # Path to conda installation
# Function to create conda activation command
get_conda_activation() {
    echo "export RAY_TMPDIR='${RAY_TMPDIR}' && source ${CONDA_PATH}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV}"

}


# Function to set cluster configuration
set_ray_cluster() {
    local head_node=$1
    shift  # Remove first argument
    local worker_nodes=("$@")  # All remaining arguments are worker nodes
    
    RAY_HEAD_NODE="$head_node"
    RAY_WORKER_NODES=("${worker_nodes[@]}")
    ALL_NODES=("$RAY_HEAD_NODE" "${RAY_WORKER_NODES[@]}")
    TOTAL_NODES=${#ALL_NODES[@]}
    
    echo -e "${GREEN}Ray cluster configuration:${NC}"
    echo "  Head node: $RAY_HEAD_NODE"
    echo "  Worker nodes: ${RAY_WORKER_NODES[@]}"
    echo "  Total nodes: $TOTAL_NODES"
    echo "  Ray temp directory: $RAY_TMPDIR"
}

# Function to ensure RAY_TMPDIR exists on a node
ensure_ray_tmpdir() {
    local node=$1
    local is_local=false
    
    # Check if this is the local node
    if [[ "$(hostname -I | awk '{print $1}')" == "$node" ]] || [[ "$(hostname)" == "$node" ]]; then
        is_local=true
    fi
    
    if [ "$is_local" = true ]; then
        mkdir -p "${RAY_TMPDIR}"
    else
        ssh ${RAY_SSH_USER}@${node} "mkdir -p '${RAY_TMPDIR}'"
    fi
}

# Function to check if Ray is running on a node with activation
check_ray_on_node() {
    local node=$1
    local is_local=false
    
    # Check if this is the local node
    if [[ "$(hostname -I | awk '{print $1}')" == "$node" ]] || [[ "$(hostname)" == "$node" ]]; then
        is_local=true
    fi
    
    if [ "$is_local" = true ]; then
        bash -c "$(get_activation_command) && ray status" &>/dev/null
    else
        ssh -o ConnectTimeout=5 ${RAY_SSH_USER}@${node} "bash -c '$(get_activation_command) && ray status'" &>/dev/null
    fi
    return $?
}

# Function to run command on node with activation
run_on_node() {
    local node=$1
    local cmd=$2
    local is_local=false
    
    # Always wrap command with activation
    local full_cmd="$(get_activation_command) && ${cmd}"
    
    # Check if this is the local node
    if [[ "$(hostname -I | awk '{print $1}')" == "$node" ]] || [[ "$(hostname)" == "$node" ]]; then
        is_local=true
    fi
    
    if [ "$is_local" = true ]; then
        bash -c "$full_cmd"
    else
        ssh ${RAY_SSH_USER}@${node} "bash -c '$full_cmd'"
    fi
}

# Function to run command on node in background with activation
run_on_node_bg() {
    local node=$1
    local cmd=$2
    
    # Always wrap command with activation
    local full_cmd="$(get_activation_command) && ${cmd}"
    
    # Check if this is the local node
    if [[ "$(hostname -I | awk '{print $1}')" == "$node" ]] || [[ "$(hostname)" == "$node" ]]; then
        bash -c "$full_cmd" &
    else
        ssh ${RAY_SSH_USER}@${node} "bash -c '$full_cmd'" &
    fi
}

# Function to start Ray cluster
start_ray_cluster() {
    if [ -z "$RAY_HEAD_NODE" ]; then
        echo -e "${RED}Error: No cluster configuration set. Call set_ray_cluster first.${NC}"
        return 1
    fi
    
    echo -e "${GREEN}Starting Ray cluster with $TOTAL_NODES nodes...${NC}"
    echo "  Using activation script: ${ACTIVATION_PATH}"
    echo "  Using Ray temp directory: ${RAY_TMPDIR}"
    
    # Ensure RAY_TMPDIR exists on all nodes
    echo -e "${YELLOW}Creating Ray temp directories on all nodes...${NC}"
    for node in "${ALL_NODES[@]}"; do
        ensure_ray_tmpdir $node
    done
    
    # Check if cluster is already running
    if check_ray_on_node $RAY_HEAD_NODE; then
        echo -e "${YELLOW}Warning: Ray is already running on head node${NC}"
        echo "Use 'stop_ray_cluster' first if you want to restart"
        return 1
    fi
    
    # Start head node with RAY_TMPDIR
    echo -e "${YELLOW}Starting head node on $RAY_HEAD_NODE...${NC}"
    run_on_node $RAY_HEAD_NODE "ray start --head --port=$RAY_PORT --dashboard-host=0.0.0.0 --dashboard-port=$RAY_DASHBOARD_PORT --temp-dir=$RAY_TMPDIR"
    
    # Wait for head node to be ready
    sleep 5
    
    # Start all worker nodes in parallel with RAY_TMPDIR
    if [ ${#RAY_WORKER_NODES[@]} -gt 0 ]; then
        for node in "${RAY_WORKER_NODES[@]}"; do
            echo -e "${YELLOW}Starting worker node on $node...${NC}"
            run_on_node_bg $node "ray start --address='${RAY_HEAD_NODE}:${RAY_PORT}' --temp-dir=$RAY_TMPDIR"
        done
        
        # Wait for all background jobs
        wait
    fi
    
    # Give cluster time to stabilize
    sleep 5
    
    # Verify cluster status
    echo -e "${GREEN}Verifying cluster status...${NC}"
    run_on_node $RAY_HEAD_NODE "ray status"
    
    echo -e "${GREEN}Ray cluster started successfully!${NC}"
    echo -e "Dashboard available at: http://${RAY_HEAD_NODE}:${RAY_DASHBOARD_PORT}"
}

# Function to stop Ray cluster
stop_ray_cluster() {
    if [ -z "$RAY_HEAD_NODE" ]; then
        echo -e "${RED}Error: No cluster configuration set.${NC}"
        return 1
    fi
    
    echo -e "${RED}Stopping Ray cluster ($TOTAL_NODES nodes)...${NC}"
    
    # Stop all nodes in parallel
    for node in "${ALL_NODES[@]}"; do
        echo -e "${YELLOW}Stopping Ray on $node...${NC}"
        run_on_node_bg $node "ray stop --force"
    done
    
    # Wait for all stops to complete
    wait
    
    echo -e "${GREEN}Ray cluster stopped.${NC}"
}

# Function to restart Ray cluster
restart_ray_cluster() {
    stop_ray_cluster
    sleep 2
    start_ray_cluster
}

# Function to check cluster status
check_ray_cluster() {
    if [ -z "$RAY_HEAD_NODE" ]; then
        echo -e "${RED}Error: No cluster configuration set.${NC}"
        return 1
    fi
    
    echo -e "${GREEN}Checking Ray cluster status ($TOTAL_NODES nodes)...${NC}"
    echo "  Activation script: ${ACTIVATION_PATH}"
    echo "  Ray temp directory: ${RAY_TMPDIR}"
    
    local all_running=true
    
    # Check head node
    echo -ne "Head node ($RAY_HEAD_NODE): "
    if check_ray_on_node $RAY_HEAD_NODE; then
        echo -e "${GREEN}✓ Running${NC}"
    else
        echo -e "${RED}✗ Not running${NC}"
        all_running=false
    fi
    
    # Check all worker nodes
    for node in "${RAY_WORKER_NODES[@]}"; do
        echo -ne "Worker node ($node): "
        if check_ray_on_node $node; then
            echo -e "${GREEN}✓ Running${NC}"
        else
            echo -e "${RED}✗ Not running${NC}"
            all_running=false
        fi
    done
    
    # Show detailed status if cluster is running
    if [ "$all_running" = true ]; then
        echo -e "\n${GREEN}Detailed cluster status:${NC}"
        run_on_node $RAY_HEAD_NODE "ray status"
        return 0
    else
        return 1
    fi
}

# Function to clean up Ray
cleanup_ray_cluster() {
    if [ -z "$RAY_HEAD_NODE" ]; then
        echo -e "${RED}Error: No cluster configuration set.${NC}"
        return 1
    fi
    
    echo -e "${RED}Cleaning up Ray cluster...${NC}"
    
    # First stop Ray
    stop_ray_cluster
    
    # Then clean up temp files in both default and custom locations
    echo -e "${YELLOW}Removing temporary files on $TOTAL_NODES nodes...${NC}"
    for node in "${ALL_NODES[@]}"; do
        echo "Cleaning $node..."
        if [[ "$(hostname -I | awk '{print $1}')" == "$node" ]] || [[ "$(hostname)" == "$node" ]]; then
            sudo rm -rf /tmp/ray/* &
            rm -rf "${RAY_TMPDIR}"/* &
        else
            ssh ${RAY_SSH_USER}@${node} "sudo rm -rf /tmp/ray/* && rm -rf '${RAY_TMPDIR}'/*" &
        fi
    done
    
    wait
    echo -e "${GREEN}Cleanup completed.${NC}"
}

# Function to get Ray address for training
get_ray_address() {
    echo "ray://${RAY_HEAD_NODE}:10001"
}

# Function to wait for cluster to be ready
wait_for_ray_cluster() {
    local max_attempts=30
    local attempt=0
    
    echo -e "${YELLOW}Waiting for Ray cluster to be ready...${NC}"
    
    while [ $attempt -lt $max_attempts ]; do
        if check_ray_on_node $RAY_HEAD_NODE; then
            echo -e "${GREEN}Ray cluster is ready!${NC}"
            return 0
        fi
        
        echo -n "."
        sleep 2
        ((attempt++))
    done
    
    echo -e "\n${RED}Timeout waiting for Ray cluster${NC}"
    return 1
}

# Function to test Ray installation on all nodes
test_ray_installation() {
    if [ -z "$RAY_HEAD_NODE" ]; then
        echo -e "${RED}Error: No cluster configuration set.${NC}"
        return 1
    fi
    
    echo -e "${GREEN}Testing Ray installation on all nodes...${NC}"
    echo "  Activation script: ${ACTIVATION_PATH}"
    echo "  Ray temp directory: ${RAY_TMPDIR}"
    echo ""
    
    local all_good=true
    
    for node in "${ALL_NODES[@]}"; do
        echo -ne "Testing $node: "
        if run_on_node $node "ray --version" &>/dev/null; then
            local version=$(run_on_node $node "ray --version 2>&1 | grep -oP 'ray, version \K[\d.]+'")
            echo -e "${GREEN}✓ Ray ${version} found${NC}"
        else
            echo -e "${RED}✗ Ray not found or not accessible${NC}"
            all_good=false
            # Try to get more info
            echo "  Debugging info:"
            run_on_node $node "which python"
            run_on_node $node "which ray || echo 'Ray not in PATH'"
            run_on_node $node "echo 'Current environment path: \$PATH'"
        fi
    done
    
    if [ "$all_good" = true ]; then
        echo -e "\n${GREEN}All nodes have Ray installed and accessible!${NC}"
        return 0
    else
        echo -e "\n${RED}Some nodes are missing Ray installation.${NC}"
        echo "Please ensure Ray is installed and accessible via '${ACTIVATION_PATH}' on all nodes."
        return 1
    fi
}

# Function to run custom command on all nodes
run_on_all_nodes() {
    local cmd=$1
    echo -e "${GREEN}Running command on all nodes: ${cmd}${NC}"
    
    for node in "${ALL_NODES[@]}"; do
        echo -e "${YELLOW}Node $node:${NC}"
        run_on_node $node "$cmd"
        echo ""
    done
}

# Function to check environment on all nodes
check_environment() {
    echo -e "${GREEN}Checking environment on all nodes...${NC}"
    
    for node in "${ALL_NODES[@]}"; do
        echo -e "${YELLOW}Node $node:${NC}"
        run_on_node $node "echo 'Python path:' && which python && echo 'Python version:' && python --version && echo 'Activation script:' && echo '${ACTIVATION_PATH}' && echo 'Ray temp dir:' && echo '${RAY_TMPDIR}'"
        echo ""
    done
}

# Export functions
export -f set_ray_cluster
export -f start_ray_cluster
export -f stop_ray_cluster
export -f restart_ray_cluster
export -f check_ray_cluster
export -f cleanup_ray_cluster
export -f get_ray_address
export -f wait_for_ray_cluster
export -f test_ray_installation
export -f check_ray_on_node
export -f run_on_node
export -f run_on_node_bg
export -f run_on_all_nodes
export -f check_environment
export -f get_activation_command
export -f ensure_ray_tmpdir