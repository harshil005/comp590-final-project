#!/bin/bash
# AWS App Runner Deployment Script
# This script builds the Docker image and pushes it to ECR

# Set variables - UPDATE THESE FOR YOUR ENVIRONMENT
REGION="us-east-1"  # Change to your AWS region
REPO_NAME="options-app"
SERVICE_NAME="options-app-service"

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

if [ -z "$ACCOUNT_ID" ]; then
    echo "Error: Could not get AWS account ID. Make sure AWS CLI is configured."
    exit 1
fi

echo "Account ID: $ACCOUNT_ID"
echo "Region: $REGION"
echo "Repository: $REPO_NAME"

# Build Docker image
echo "Building Docker image..."
docker build -t $REPO_NAME .

if [ $? -ne 0 ]; then
    echo "Error: Docker build failed"
    exit 1
fi

# Login to ECR
echo "Logging into ECR..."
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

if [ $? -ne 0 ]; then
    echo "Error: ECR login failed"
    exit 1
fi

# Check if repository exists, create if it doesn't
echo "Checking ECR repository..."
aws ecr describe-repositories --repository-names $REPO_NAME --region $REGION > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Creating ECR repository..."
    aws ecr create-repository --repository-name $REPO_NAME --region $REGION
fi

# Tag image
echo "Tagging image..."
docker tag $REPO_NAME:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:latest

# Push to ECR
echo "Pushing image to ECR..."
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:latest

if [ $? -ne 0 ]; then
    echo "Error: Docker push failed"
    exit 1
fi

echo ""
echo "Deployment complete!"
echo "Image URI: $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:latest"
echo ""
echo "Next steps:"
echo "1. Go to AWS App Runner console"
echo "2. Create or update service with the image URI above"
echo "3. Configure port 8080, environment variables, and instance settings"
echo "4. Deploy and monitor"

