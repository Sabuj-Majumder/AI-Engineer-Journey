from collections import Counter
import onnx
from src.config import RESULTS_DIR, ONNX_PATH

paths = [ONNX_PATH] + sorted((RESULTS_DIR / "graphs").glob("*.onnx"))
for p in paths:
    g = onnx.load(str(p)).graph
    ops = Counter(n.op_type for n in g.node)
    print(f"{p.name}: {len(g.node)} nodes | top ops: {dict(ops.most_common(6))}")