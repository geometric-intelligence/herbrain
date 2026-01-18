#!/usr/bin/env python3
"""
Generate metadata JSON mapping gestational weeks to MRI session files.

This creates a metadata.json file that the frontend uses to determine
which NIfTI file to load for a given gestational week.

Usage:
    python generate_metadata.py --data-dir /path/to/data --output ../public/data/metadata.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Generate week-to-session metadata for MRI files"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.environ.get("HERBRAIN_DATA_DIR", "~/.herbrain/data/"),
        help="Directory containing herbrain data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../public/data/metadata.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--r2-base-url",
        type=str,
        default="https://herbrain-mri.r2.cloudflarestorage.com",
        help="Base URL for R2 bucket",
    )
    
    args = parser.parse_args()
    
    # Expand user path
    data_dir = os.path.expanduser(args.data_dir)
    pregnancy_dir = os.path.join(data_dir, "pregnancy")
    
    # Try to load hormone data for week mapping
    csv_path = os.path.join(pregnancy_dir, "raw", "28Baby_Hormones.csv")
    
    metadata = {
        "weekToSession": {},
        "sessionUrls": {},
    }
    
    if os.path.exists(csv_path):
        print(f"Loading hormone data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Extract session ID from sessionID column
        if "sessionID" in df.columns:
            df["session_num"] = df["sessionID"].apply(
                lambda x: int(x.split("-")[1]) if isinstance(x, str) and "-" in x else x
            )
        else:
            df["session_num"] = df.index + 1
        
        # Map each gestational week to closest session
        if "gestWeek" in df.columns:
            gest_weeks = df["gestWeek"].values
            session_nums = df["session_num"].values
            
            for week in range(0, 46):
                closest_idx = abs(gest_weeks - week).argmin()
                session = int(session_nums[closest_idx])
                session_id = f"session-{session:02d}"
                
                metadata["weekToSession"][week] = session_id
                metadata["sessionUrls"][session_id] = f"{args.r2_base_url}/subject-01/{session_id}.nii.gz"
    else:
        print(f"Warning: Could not find hormone data at {csv_path}")
        print("Generating placeholder metadata...")
        
        # Generate placeholder mapping (every 5 weeks)
        for week in range(0, 41, 5):
            session_id = f"session-{(week // 5) + 1:02d}"
            metadata["weekToSession"][week] = session_id
            metadata["sessionUrls"][session_id] = f"{args.r2_base_url}/subject-01/{session_id}.nii.gz"
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to JSON
    print(f"Writing metadata to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Done! Created metadata for {len(metadata['weekToSession'])} weeks")


if __name__ == "__main__":
    main()
