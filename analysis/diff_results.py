import polars as pl

table_70 = pl.read_csv("results/table_70.csv")
table_90 = pl.read_csv("results/table_90.csv")

def compute_differences(df: pl.DataFrame) -> pl.DataFrame:
    models = [c for c in df.columns if c not in ("dir", "radial")]
    exprs = [pl.col(m) - pl.col("radial") for m in models]
    exprs = [e * -1 for e in exprs]
    diff_df = df.select(["dir"] + exprs)
    rename_map = {m: f"{m}_minus_radial" for m in models}
    diff_df = diff_df.rename(rename_map)
    return diff_df

diff_70 = compute_differences(table_70)
diff_90 = compute_differences(table_90)

print("=== Differences (70) ===")
print(diff_70)

print("\n=== Differences (90) ===")
print(diff_90)

diff_70.write_csv("results/diff_70.csv")
diff_90.write_csv("results/diff_90.csv")

