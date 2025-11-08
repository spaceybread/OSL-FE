import polars as pl
import numpy as np
from pathlib import Path

root_dirs = ["voice_128", "voice_64", "ytf_128", "ytf_64", "vggface2_64", "vggface2_128"]

zero_map = {
    "voice_128": 16.0743087414,
    "voice_64": 16.0743087414,
    "ytf_128": 16.8690636267,
    "ytf_64": 16.8690636267,
    "vggface2_64": 18.970004751,
    "vggface2_128": 18.970004751,
}

models = ["radial", "e8", "gauss_0", "gauss_1", "gauss_2"]
temps = [70, 90]

tables = {70: {}, 90: {}}

for name in root_dirs:
    res_dir = Path("../datasets/" + name + "/tests/results")
    for temp in temps:
        row = {}
        for model in models:
            file = res_dir / f"{model}_{temp}.csv"
            if not file.exists():
                row[model] = None
                continue
            df = pl.read_csv(file)
            fmr = df["TMR"][0]
            if fmr > 0:
                val = fmr * 100
            else:
                val = zero_map[name]
            row[model] = val
        tables[temp][name] = row

table_70 = pl.DataFrame(
    {**{"dir": list(tables[70].keys())},
     **{m: [tables[70][d][m] for d in tables[70]] for m in models}}
).set_sorted("dir")

table_90 = pl.DataFrame(
    {**{"dir": list(tables[90].keys())},
     **{m: [tables[90][d][m] for d in tables[90]] for m in models}}
).set_sorted("dir")

print("=== Table 70 ===")
print(table_70)

table_70.write_csv("results/tmr_70.csv")

print("\n=== Table 90 ===")
print(table_90)

table_90.write_csv("results/tmr_90.csv")
