import pandas as pd
import sys
# Load CSV
df = pd.read_csv(sys.argv[1])

# Round values
df["TMR"] = df["TMR"].round(5)
df["FMR"] = df["FMR"].round(5)

df["outer"] = df["outer"].map(lambda x: f"{x:.3f}")
df["radius"] = df["radius"].map(lambda x: f"{x:.3f}")

# Create cell content "(TMR, FMR)"
df["cell"] = df.apply(lambda r: f"({r.TMR:.5f}, {r.FMR:.5f})", axis=1)

# Pivot table
table = df.pivot(index="outer", columns="radius", values="cell")

# Sort axes (optional but nice)
table = table.sort_index().sort_index(axis=1)

print(table)

# Generate LaTeX tabular
latex_tabular = table.to_latex(
    escape=False,
    column_format="l|" + "c" * len(table.columns),
    multicolumn=True,
    multicolumn_format="c",
    bold_rows=False,
    index_names=False
)

# Wrap in full table environment
latex_table = rf"""
\begin{{table}}[t]
    \caption{{TMR and FMR as a function of outer and radius}}
    \label{{tab:outer-radius}}
    \centering\tabfontsize
{latex_tabular}
\end{{table}}
"""

# print(latex_table)

