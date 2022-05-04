# The 2stems ckpt directory.
mkdir 2stems

# Download and extract the 2stems ckpt.
wget https://github.com/deezer/spleeter/releases/download/v1.4.0/2stems.tar.gz
tar -xvzf 2stems.tar.gz

# Move extracted files to 2stems directory.
mv checkpoint 2stems/
mv model.data-00000-of-00001 2stems/
mv model.index 2stems/
mv model.meta 2stems/

# Remove the 2stems.tar.gz file.
rm 2stems.tar.gz