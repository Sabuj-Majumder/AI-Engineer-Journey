import json
import pandas as pd
from src.config import RESULTS_DIR

rows = []
for f in sorted(RESULTS_DIR.glob("bench_*.json")):
    name = f.stem.removeprefix("bench_")
    for r in json.loads(f.read_text()):
        rows.append({"config": name, **r})
df = pd.DataFrame(rows)
base = df[df.config == "pytorch"].set_index("batch_size")["p50_ms"]
df["speedup_vs_pytorch"] = df.apply(lambda r: base[r.batch_size] / r.p50_ms, axis=1)
cols = ["config", "batch_size", "p50_ms", "p95_ms", "p99_ms",
        "throughput_samples_per_sec", "speedup_vs_pytorch"]
md = df[cols].round(3).sort_values(["batch_size", "config"]).to_markdown(index=False)
(RESULTS_DIR / "summary.md").write_text(md)
print(md)
print()
for f in sorted(RESULTS_DIR.glob("parity_*.json")):
    print(f.read_text())