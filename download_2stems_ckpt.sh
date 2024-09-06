#!/bin/bash

# This script downloads and sets up the 2stems checkpoint for the Spleeter model.

# Create directory for 2stems checkpoint
mkdir 2stems

# Download the 2stems checkpoint archive
wget https://github.com/deezer/spleeter/releases/download/v1.4.0/2stems.tar.gz

# Extract the contents of the archive
tar -xvzf 2stems.tar.gz

# Move extracted checkpoint files to the 2stems directory
mv checkpoint 2stems/
mv model.data-00000-of-00001 2stems/
mv model.index 2stems/
mv model.meta 2stems/

# Clean up by removing the downloaded archive
rm 2stems.tar.gz