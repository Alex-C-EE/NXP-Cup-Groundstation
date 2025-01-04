#!/usr/bin/env python3

import csv
import sys
from pathlib import Path

def fix_timestamps(input_file: str, output_file: str = None):
    """
    Read CSV file and replace first column with incremental timestamps.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file. If None, will use input_file_fixed.csv
    """
    # Handle output filename
    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}_fixed{input_path.suffix}")
    
    try:
        # Read input file and write to output
        with open(input_file, 'r') as fin, open(output_file, 'w', newline='') as fout:
            reader = csv.reader(fin)
            writer = csv.writer(fout)
            
            # Copy header
            header = next(reader)
            writer.writerow(header)
            
            # Process data rows
            timestamp = 0
            for row in reader:
                # Replace first column with incremental timestamp
                row[0] = str(timestamp)
                writer.writerow(row)
                timestamp += 100  # Increment by 100ms
                
        print(f"Successfully processed file. Output saved to: {output_file}")
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_timestamps.py input.csv [output.csv]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    fix_timestamps(input_file, output_file)