#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "=========================================================="
echo "🚀 Multi-Agent Financial Intelligence Platform AWS Deploy"
echo "=========================================================="

# 1. Check OS distribution
OS_TYPE=$(cat /etc/os-release | grep -E '^ID=' | cut -d'=' -f2 | tr -d '"')
echo "Detected OS: $OS_TYPE"

# Function to install Docker on Ubuntu
install_docker_ubuntu() {
    echo "Installing Docker on Ubuntu..."
    sudo apt-get update
    sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common gnupg lsb-release git
    
    # Add Docker's official GPG key
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg || true
    
    # Set up the repository
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
      
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
}

# Function to install Docker on Amazon Linux
install_docker_amazon_linux() {
    echo "Installing Docker on Amazon Linux..."
    sudo dnf update -y || sudo yum update -y
    sudo dnf install -y docker git || sudo yum install -y docker git
    sudo systemctl start docker
    sudo systemctl enable docker
    
    # Install Docker Compose manually for Amazon Linux if docker-compose-plugin isn't packaged
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    sudo ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose || true
}

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    if [ "$OS_TYPE" = "ubuntu" ]; then
        install_docker_ubuntu
    elif [ "$OS_TYPE" = "amzn" ] || [ "$OS_TYPE" = "rhel" ] || [ "$OS_TYPE" = "centos" ]; then
        install_docker_amazon_linux
    else
        echo "Unsupported OS for auto-docker installation. Please install docker manually."
        exit 1
    fi
else
    echo "✔ Docker is already installed."
fi

# Ensure docker service is running
sudo systemctl start docker || true
sudo systemctl enable docker || true

# Add current user to docker group
if ! groups $USER | grep &>/dev/null '\bdocker\b'; then
    echo "Adding $USER to the docker group..."
    sudo usermod -aG docker $USER
    echo "⚠️ You may need to log out and log back in (or run 'newgrp docker') for docker permissions to apply without sudo."
fi

# Determine compose command
if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
elif command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
else
    # Install docker compose plugin if missing on Ubuntu
    if [ "$OS_TYPE" = "ubuntu" ]; then
        sudo apt-get install -y docker-compose-plugin
        COMPOSE_CMD="docker compose"
    else
        echo "Docker Compose not found. Please install it."
        exit 1
    fi
fi
echo "Using docker compose command: $COMPOSE_CMD"

# 2. Clone or Update Repository
REPO_DIR="Multi-Agent-Financial-Intelligence-Platform"
REPO_URL="https://github.com/ohm2905/Multi-Agent-Financial-Intelligence-Platform.git"

if [ -d "$REPO_DIR" ]; then
    echo "Directory $REPO_DIR exists. Pulling latest updates..."
    cd "$REPO_DIR"
    git pull
else
    echo "Cloning repository..."
    git clone "$REPO_URL"
    cd "$REPO_DIR"
fi

# 3. Setup Environment Variables
ENV_FILE="backend/.env"
mkdir -p backend

echo "=========================================================="
echo "🔧 Configuring Environment Variables"
echo "=========================================================="

# Helper function to set/overwrite environment variables
set_env_val() {
    local var_name=$1
    local var_value=$2
    if [ -f "$ENV_FILE" ]; then
        # Remove existing key
        grep -v "^${var_name}=" "$ENV_FILE" > "${ENV_FILE}.tmp" || true
        mv "${ENV_FILE}.tmp" "$ENV_FILE"
    fi
    echo "${var_name}=${var_value}" >> "$ENV_FILE"
}

# Helper function to prompt for variable if not exists
get_env_val() {
    local var_name=$1
    local prompt_msg=$2
    local current_val=""
    
    if [ -f "$ENV_FILE" ]; then
        current_val=$(grep -E "^${var_name}=" "$ENV_FILE" | cut -d'=' -f2-)
    fi
    
    if [ -n "$current_val" ]; then
        read -p "$prompt_msg [$current_val]: " input_val
        if [ -z "$input_val" ]; then
            input_val=$current_val
        fi
    else
        read -p "$prompt_msg: " input_val
    fi
    
    set_env_val "$var_name" "$input_val"
}

# Set defaults for Docker stack
set_env_val "DATABASE_URL" "postgresql://postgres:postgres_password@db:5432/financial_platform"
set_env_val "REDIS_HOST" "redis"
set_env_val "REDIS_PORT" "6379"
set_env_val "HF_HOME" "/app/data/cache/huggingface"
set_env_val "SENTENCE_TRANSFORMERS_HOME" "/app/data/cache/sentence_transformers"
set_env_val "CHROMA_DB_DIR" "/app/data/chromadb"

# Get API Keys
get_env_val "GEMINI_API_KEY" "Enter your Gemini API Key"
get_env_val "TAVILY_API_KEY" "Enter your Tavily API Key"
get_env_val "NEWS_API_KEY" "Enter your News API Key"

echo "✔ Environment configured in $ENV_FILE"

# 4. Spin up Docker Stack
echo "=========================================================="
echo "🚀 Booting up the containers using $COMPOSE_CMD..."
echo "=========================================================="
sudo $COMPOSE_CMD up -d --build

echo "=========================================================="
echo "🎉 Deployment initiated successfully!"
echo "=========================================================="
echo "The platform is running on port 8000."
echo "You can check status using:"
echo "  sudo docker ps"
echo "  sudo docker logs -f financial_backend"
echo "=========================================================="
