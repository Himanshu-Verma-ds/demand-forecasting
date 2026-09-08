"""Build a small subset of the real data plus a fast config so every stage can run end to end."""
import pathlib

import pandas as pd
import yaml

RAW = "data/raw/data.csv"
OUT_DIR = pathlib.Path("_audit")
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(RAW, parse_dates=["date"])

# Keep 20 complete store-SKU series so lookback(56)+horizon(14) windows exist.
counts = df.groupby(["store_id", "sku_id"]).size()
full_series = counts[counts == counts.max()].index[:20]
keep = pd.MultiIndex.from_tuples(list(full_series), names=["store_id", "sku_id"])

sub = df.set_index(["store_id", "sku_id"]).loc[keep].reset_index()
sub = sub[df.columns]
sub.to_csv(OUT_DIR / "data.csv", index=False)

cfg = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
cfg["data"]["raw_path"] = "_audit/data.csv"
cfg["training"]["save_dir"] = "_audit/artifacts"
cfg["training"]["mlflow_experiment"] = "audit-smoke"
cfg["bayes"]["n_trials"] = 2
cfg["dl"]["epochs"] = 2
cfg["dl"]["patience"] = 1
cfg["dl"]["num_workers"] = 0
cfg["logging"]["dir"] = "_audit/logs"
cfg["logging"]["console"] = False

yaml.safe_dump(cfg, open(OUT_DIR / "config.yaml", "w", encoding="utf-8"), sort_keys=False)

print("rows", len(sub), "series", sub.groupby(['store_id', 'sku_id']).ngroups)
print("dates", sub["date"].min().date(), "->", sub["date"].max().date())
