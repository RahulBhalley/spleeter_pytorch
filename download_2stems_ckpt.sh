#!/bin/bash

# This script downloads and sets up the 2stems checkpoint for the Spleeter model.
# It creates a directory, downloads the checkpoint, extracts it, and cleans up.

# Set variables
CHECKPOINT_DIR="2stems"
CHECKPOINT_URL="https://github.com/deezer/spleeter/releases/download/v1.4.0/2stems.tar.gz"
ARCHIVE_NAME="2stems.tar.gz"

# Create directory for 2stems checkpoint
mkdir -p "$CHECKPOINT_DIR"

# Download the 2stems checkpoint archive
echo "Downloading 2stems checkpoint..."
wget -q --show-progress "$CHECKPOINT_URL" -O "$ARCHIVE_NAME"

# Extract the contents of the archive
echo "Extracting checkpoint files..."
tar -xzf "$ARCHIVE_NAME" -C "$CHECKPOINT_DIR"

# Clean up by removing the downloaded archive
echo "Cleaning up..."
rm "$ARCHIVE_NAME"

echo "2stems checkpoint setup complete. Files are in the '$CHECKPOINT_DIR' directory."