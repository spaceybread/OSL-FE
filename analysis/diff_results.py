import polars as pl

table_70 = pl.read_csv("results/table_70.csv")
table_90 = pl.read_csv("results/table_90.csv")

def compute_differences(df: pl.DataFrame) -> pl.DataFrame:
    models = [c for c in df.columns if c not in ("dir", "radial")]

    exprs = [((pl.col("radial") - pl.col(m))).alias(f"rad_minus_{m}") for m in models]

    diff_df = df.select(["dir", "radial"] + exprs)

    return diff_df

diff_70 = compute_differences(table_70)
diff_90 = compute_differences(table_90)

print("=== Differences (70) ===")
print(diff_70)

print("\n=== Differences (90) ===")
print(diff_90)

diff_70.write_csv("results/diff_70.csv")
diff_90.write_csv("results/diff_90.csv")

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


latex_70 = to_latex_table(diff_70, "Model differences at 70 TMR", "tab:diff70")
latex_90 = to_latex_table(diff_90, "Model differences at 90 TMR", "tab:diff90")

with open("results/diff_70_table.tex", "w") as f: f.write(latex_70)

with open("results/diff_90_table.tex", "w") as f: f.write(latex_90)

print("LaTeX tables written to results/diff_70_table.tex and results/diff_90_table.tex")
