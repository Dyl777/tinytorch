import sys
import struct
import pandas as pd
import numpy as np

def parquet_to_binary(parquet_path, output_path):
    df = pd.read_parquet(parquet_path)
    
    numeric_cols = df.select_dtypes(include=[np.float32, np.float64, np.int32, np.int64]).columns.tolist()
    
    if not numeric_cols:
        print("ERROR: No numeric columns found", file=sys.stderr)
        sys.exit(1)
    
    data = df[numeric_cols].astype(np.float32).values
    
    n_rows, n_cols = data.shape
    
    with open(output_path, 'wb') as f:
        # Magic header
        f.write(b'PQTD')
        # Use fixed-size 64-bit integers for cross-platform compatibility
        f.write(struct.pack('<Q', n_rows))  # uint64 little-endian
        f.write(struct.pack('<Q', n_cols))  # uint64 little-endian
        
        # Column names
        for col in numeric_cols:
            col_bytes = col.encode('utf-8')
            f.write(struct.pack('<I', len(col_bytes)))  # uint32 for name length
            f.write(col_bytes)
        
        # Data as float32
        f.write(data.tobytes())
    
    print(f"Converted {parquet_path} to {output_path}: {n_rows} rows x {n_cols} cols")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.parquet> <output.bin>", file=sys.stderr)
        sys.exit(1)
    
    parquet_to_binary(sys.argv[1], sys.argv[2])
