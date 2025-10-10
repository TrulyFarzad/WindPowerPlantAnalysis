
import sys
import importlib.util
from pathlib import Path
import yaml
import copy

def find_project_root(user_project_root: str = None):
    if user_project_root:
        p = Path(user_project_root).expanduser().resolve()
        if (p / "main.py").exists():
            return str(p)
        raise FileNotFoundError(f"main.py not found under provided path: {p}")
    here = Path.cwd().resolve()
    for cand in [here] + list(here.parents):
        if (cand / "main.py").exists():
            return str(cand)
    raise FileNotFoundError("Could not find 'main.py'. Please set PROJECT_ROOT in the sidebar.")

def load_default_config(project_root: str):
    cfg_path = Path(project_root) / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    # minimal
    return {
        "project": {"seed": 4, "years": 10, "discount_rate": 0.12},
        "wind": {"city": "Khaf", "hub_height_m": 100, "alpha": 0.14},
        "plant": {"capacity_mw": 20.0},
        "price": {},
        "costs": {"capex_usd": 20000000, "opex_usd_per_kw_yr": 40, "escalation": 0.20},
        "monte_carlo": {"iterations": 2000, "random_seed": 4},
        "report": {"out_html": "mvp_report.html"}
    }

def make_cfg_for_run(default_cfg: dict,
                     city: str,
                     years: int,
                     capacity_mw: float,
                     discount_rate: float,
                     capex_per_kw: float,
                     opex_per_kw_yr: float,
                     inflation_opex: float,
                     turbine_source: str,
                     turbine_name: str,
                     iec_class: str,
                     n_scenarios: int):
    cfg = copy.deepcopy(default_cfg if isinstance(default_cfg, dict) else {})

    w = cfg.setdefault("wind", {})
    w["city"] = city
    w["turbine_source"] = turbine_source
    w["turbine_name"] = turbine_name
    w["iec_class"] = iec_class

    p = cfg.setdefault("plant", {})
    p["capacity_mw"] = float(capacity_mw)

    proj = cfg.setdefault("project", {})
    proj["years"] = int(years)
    proj["discount_rate"] = float(discount_rate)

    mc = cfg.setdefault("monte_carlo", {})
    mc["iterations"] = int(n_scenarios)
    mc.setdefault("random_seed", cfg.get("project", {}).get("seed", 4))

    costs = cfg.setdefault("costs", {})
    total_capex = float(capacity_mw) * 1000.0 * float(capex_per_kw)
    costs["capex_usd"] = total_capex
    costs["opex_usd_per_kw_yr"] = float(opex_per_kw_yr)
    costs["escalation"] = float(inflation_opex)

    return cfg

def _import_main_from(project_root: str):
    project_root = str(Path(project_root).resolve())
    main_file = Path(project_root) / "main.py"
    if not main_file.exists():
        raise FileNotFoundError(f"main.py not found in {project_root}")

    # ensure 'src' imports work
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    spec = importlib.util.spec_from_file_location("project_main", str(main_file))
    if not spec or not spec.loader:
        raise ImportError("Could not build import spec for main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def run_project_with_config(project_root: str, cfg: dict):
    mod = _import_main_from(project_root)
    if not hasattr(mod, "run"):
        raise AttributeError("main.py must expose run(cfg: dict, out_html: str = 'mvp_report.html')")
    out_html = (cfg.get("report") or {}).get("out_html") or "mvp_report.html"
    mod.run(cfg=cfg, out_html=out_html)
    out_path = Path(project_root) / out_html
    if not out_path.exists():
        cands = sorted(Path(project_root).glob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands:
            raise FileNotFoundError("Pipeline finished but HTML report was not found.")
        return str(cands[0])
    return str(out_path)
