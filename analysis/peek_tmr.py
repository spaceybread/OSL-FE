import polars as pl
import numpy as np
from pathlib import Path

def to_latex_table(df: pl.DataFrame, caption: str, label: str) -> str:
    # Find numeric columns
    float_cols = [c for c, dtype in zip(df.columns, df.dtypes) if dtype in (pl.Float32, pl.Float64)]

    # Round only numeric columns
    df = df.with_columns([
        pl.col(c).round(2).alias(c) for c in float_cols
    ])

    # Convert to pandas for LaTeX export
    pdf = df.to_pandas()

    latex_str = pdf.to_latex(
        index=False,
        caption=caption,
        label=label,
        float_format="%.2f",
        column_format="l" + "c" * (len(df.columns) - 1),
        escape=False
    )

    return latex_str


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

latex_70 = to_latex_table(table_70, "Testing with target 70 TMR", "tab:tmr70")
latex_90 = to_latex_table(table_90, "Testing with target 90 TMR", "tab:tmr90")
with open("results/tmr_70_table.tex", "w") as f: f.write(latex_70)
with open("results/tmr_90_table.tex", "w") as f: f.write(latex_90)
