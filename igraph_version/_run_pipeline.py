"""Standalone script: regenerate generation_pipeline.png from cache only."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _p in (_ROOT, _ROOT / "simclr", _ROOT / "diffusion", _ROOT / "generation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matplotlib
matplotlib.use("Agg")

# Read the source with explicit UTF-8 so box-drawing chars don't break cp1252
_src = Path("viz_real_generated.py").read_text(encoding="utf-8")

# Strip out generate_with_labels and everything after so no igraph encoder
# import fires at module load time.  Inject __file__ so path setup inside works.
_cutoff = _src.find("def main()")
_globals = {"__file__": str(_ROOT / "viz_real_generated.py")}
exec(compile(_src[:_cutoff], "viz_real_generated.py", "exec"), _globals)

load_network_cache  = _globals["load_network_cache"]
load_gen_cache      = _globals["load_gen_cache"]
make_pipeline_figure = _globals["make_pipeline_figure"]

import matplotlib.pyplot as plt

networks = load_network_cache("data/networks_cache_LI-Small_Trans_v2.pkl")
gen_data = load_gen_cache("data/gen_cache_LI-Small_Trans_n500_t150_ld10_encenc_full.pkl")

out_dir = Path("figures_real")
out_dir.mkdir(exist_ok=True)

# gen_labeled=None -> falls back to betweenness heuristic on gen_data
make_pipeline_figure(networks, gen_data, None, out_dir, n_rows=3, seed=7)
print("Done.")
