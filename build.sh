#!/bin/bash

# Simple build script for the technical-index project

echo "Building technical-index project..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[dev]"

# Run tests
echo "Running tests..."
pytest

# Format code
echo "Formatting code..."
black .

# Run linting
echo "Running linting..."
flake8 .

echo "Build completed successfully!"
