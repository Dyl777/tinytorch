import pandas as pd
import numpy as np

np.random.seed(42)

n_rows = 1000
data = {
    'feature_0': np.random.randn(n_rows).astype(np.float32),
    'feature_1': np.random.randn(n_rows).astype(np.float32) * 2.0 + 5.0,
    'feature_2': np.random.randn(n_rows).astype(np.float32) * 0.5 - 3.0,
    'feature_3': np.abs(np.random.randn(n_rows).astype(np.float32)),
    'label': (np.random.randn(n_rows) > 0).astype(np.float32)
}

df = pd.DataFrame(data)
df.to_parquet('sample_data.parquet', engine='pyarrow', compression='snappy')

print(f"Created sample_data.parquet with {n_rows} rows and {len(data)} columns")
print(f"Columns: {list(data.keys())}")
print(f"First 5 rows:")
print(df.head())
print(f"\nStatistics:")
print(df.describe())
