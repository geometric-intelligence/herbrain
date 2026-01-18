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

DATA_DIR="${1:-$HERBRAIN_DATA_DIR/pregnancy/raw}"
BUCKET_NAME="herbrain-mri"

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: $DATA_DIR"
    echo "Usage: $0 /path/to/mri/data"
    exit 1
fi

echo "Uploading MRI files from $DATA_DIR to R2 bucket: $BUCKET_NAME"

# Find and upload all .nii and .nii.gz files
find "$DATA_DIR" -type f \( -name "*.nii" -o -name "*.nii.gz" \) | while read -r file; do
    # Get relative path from data dir
    relative_path="${file#$DATA_DIR/}"
    
    echo "Uploading: $relative_path"
    npx --yes wrangler r2 object put "$BUCKET_NAME/$relative_path" --file="$file"
done

echo ""
echo "Upload complete!"
echo ""
echo "To enable public access (optional):"
echo "  npx wrangler r2 bucket public-access set $BUCKET_NAME --enable"
echo ""
echo "Your bucket URL will be: https://<account-id>.r2.cloudflarestorage.com/$BUCKET_NAME"
