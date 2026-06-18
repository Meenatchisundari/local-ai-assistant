import sys
from pathlib import Path
import pandas as pd

def latest_csv(out_dir=Path("data/results")):
    csvs = sorted(out_dir.glob("benchmark_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return csvs[0] if csvs else None

def summarise(csv_path):
    df = pd.read_csv(csv_path)
    ok = df[df["error"].isna()].copy()
    errs = df[df["error"].notna()]

    print(f"\n{'='*60}")
    print(f"  Results: {csv_path.name}")
    print(f"  Total runs: {len(df)}  |  OK: {len(ok)}  |  Errors: {len(errs)}")
    print(f"{'='*60}")

    if ok.empty:
        print("  No successful runs found.")
        return

    print("\n  Per-model summary\n")
    summary = ok.groupby("model").agg(
        n_runs=("run_id","count"),
        ttft_mean=("ttft_s","mean"),
        ttft_p50=("ttft_s","median"),
        ttft_p95=("ttft_s", lambda x: x.quantile(0.95)),
        tok_s_mean=("tokens_per_sec","mean"),
        tok_s_p50=("tokens_per_sec","median"),
        latency_mean=("total_latency_s","mean"),
        latency_p95=("total_latency_s", lambda x: x.quantile(0.95)),
    ).round(3)
    print(summary.to_string())

    print("\n\n  tokens/sec by model x category\n")
    pivot = ok.pivot_table(
        values="tokens_per_sec",
        index="model",
        columns="prompt_category",
        aggfunc="mean",
    ).round(1)
    print(pivot.to_string())

    if not errs.empty:
        print(f"\n\n  {len(errs)} errored runs:")
        print(errs[["model","prompt_id","repetition","error"]].to_string(index=False))

    print(f"\n  CSV: {csv_path}\n")

path = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_csv()
if path is None or not path.exists():
    print("  No benchmark CSV found. Run benchmark_harness.py first.")
    sys.exit(1)
summarise(path)
