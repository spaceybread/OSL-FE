import os
import pandas as pd
import math

base_dir = "../datasets/"

methods = ["radial", "e8", "gauss_0", "gauss_1", "gauss_2"]
datasets = []
at70 = []
at90 = []

for dataset in sorted(os.listdir(base_dir)):
    train_results = os.path.join(base_dir, dataset, "train", "results")
    if not os.path.isdir(train_results):
        continue
    
    datasets.append(dataset)
    row70, row90 = [], []
    
    for method in methods:
        file_path = os.path.join(train_results, f"{method}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if len(df) >= 2:
                fmr70 = df.loc[0, "FMR"]
                fmr90 = df.loc[1, "FMR"]
                val70 = -math.log2(fmr70) if fmr70 > 0 else None
                val90 = -math.log2(fmr90) if fmr90 > 0 else None
                row70.append(val70)
                row90.append(val90)
            else:
                row70.append(None)
                row90.append(None)
        else:
            row70.append(None)
            row90.append(None)
    
    at70.append(row70)
    at90.append(row90)

# Create DataFrames
table70 = pd.DataFrame(at70, columns=methods, index=datasets)
table90 = pd.DataFrame(at90, columns=methods, index=datasets)

print("=== Table: at 70 ===")
print(table70)
print("\n=== Table: at 90 ===")
print(table90)

