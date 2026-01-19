#!/bin/bash
# Upload MRI NIfTI files to Cloudflare R2
#
# Usage:
#   ./upload_mri.sh /path/to/mri/data
#
# Prerequisites:
#   - wrangler CLI configured with Cloudflare credentials
#   - R2 bucket 'herbrain-mri' already created

set -e

export CLOUDFLARE_ACCOUNT_ID="d02f773269d425676a8178615df5bdc5"

DATA_DIR="/home/data/pregnancy/raw/mri"
BUCKET_NAME="herbrain-mri"

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: $DATA_DIR"
    echo "Usage: $0"
    exit 1
fi

echo "Uploading BrainNormalizedToTemplate.nii.gz files from $DATA_DIR to R2 bucket: $BUCKET_NAME"

# Find and upload only BrainNormalizedToTemplate.nii.gz files in ses-* directories
find "$DATA_DIR" -type f -path "*/ses-*/BrainNormalizedToTemplate.nii.gz" | while read -r file; do
    # Get relative path from data dir
    relative_path="${file#$DATA_DIR/}"
    
    echo "Uploading: $relative_path"
    npx --yes wrangler r2 object put "$BUCKET_NAME/$relative_path" --file="$file" --remote
done

echo ""
echo "Upload complete!"
echo ""
echo "To enable public access (if not already enabled):"
echo "  npx wrangler r2 bucket dev-url enable $BUCKET_NAME"
echo ""
echo "Your bucket URL will be: https://pub-<hash>.r2.dev"
