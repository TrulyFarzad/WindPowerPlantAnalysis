# webui/backend_adapter.py
from __future__ import annotations

import os
import sys
import copy
import yaml
import types
import importlib.util
import inspect
from pathlib import Path
from contextlib import contextmanager

# --------------------------- Utilities ---------------------------

@contextmanager
def _temp_sys_path_and_cwd(path: Path):
    old_cwd = os.getcwd()
    path = Path(path).resolve()
    path_str = str(path)
    inserted = False
    try:
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
            inserted = True
        os.chdir(path_str)
        yield
    finally:
        os.chdir(old_cwd)
        if inserted and sys.path and sys.path[0] == path_str:
            sys.path.pop(0)
        else:
            try:
                sys.path.remove(path_str)
            except ValueError:
                pass

def _find_main_path(project_root: Path) -> Path:
    candidate1 = project_root / "Code" / "main.py"
    candidate2 = project_root / "main.py"
    if candidate1.exists():
        return candidate1
    if candidate2.exists():
        return candidate2
    raise FileNotFoundError(f"Could not find main.py at {candidate1} or {candidate2}")

def _import_module_from_path(mod_name: str, file_path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {mod_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module

# --------------------------- Public helpers used by app.py ---------------------------

def load_default_config(project_root: str) -> dict:
    root = Path(project_root).resolve()
    for cfg_path in [root / "Code" / "config.yaml", root / "config.yaml"]:
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    # fallback حداقلی
    return {
        "project": {"seed": 4, "years": 10, "discount_rate": 0.12},
        "wind": {"city": "Khaf", "hub_height_m": 100, "alpha": 0.14},
        "plant": {"capacity_mw": 20.0},
        "price": {},
        "costs": {"capex_usd_per_kw": 1000.0, "opex_usd_per_kw_yr": 40.0, "inflation_opex": 0.20},
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
                     n_scenarios: int) -> dict:
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

    # تنها بخش معتبر برای کد اقتصادی فعلی: costs
    costs = cfg.setdefault("costs", {})
    costs["capex_usd_per_kw"]   = float(capex_per_kw)
    costs["opex_usd_per_kw_yr"] = float(opex_per_kw_yr)
    costs["inflation_opex"]     = float(inflation_opex)

    return cfg

# --------------------------- Runner ---------------------------

def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_project_with_config(root, cfg: dict | None, out_html: str | None = None) -> str:
    """
    main.py را به‌صورت امن به‌عنوان ماژول لود می‌کند و تابع run(...) را—با هر یک از امضاهای رایج—اجرا می‌کند.
    امضاهای پشتیبانی‌شده:
      - run(cfg_path=..., out_html=...)
      - run(cfg=..., out_html=...)
      - run(cfg_path=...)
      - run(cfg=...)
    """
    root = Path(root).resolve()
    main_path = _find_main_path(root)
    code_dir = main_path.parent

    if out_html is None:
        out_html = (code_dir / "mvp_report.html").as_posix()

    default_cfg_path = code_dir / "config.yaml"
    cfg_path_to_use = default_cfg_path

    temp_cfg_path = None
    if isinstance(cfg, dict) and len(cfg) > 0:
        temp_cfg_path = code_dir / "__webui_config.yaml"
        with open(temp_cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        cfg_path_to_use = temp_cfg_path

    with _temp_sys_path_and_cwd(code_dir):
        mod = _import_module_from_path("_wp_main", main_path)
        if not hasattr(mod, "run"):
            raise RuntimeError(f"'run' function not found in {main_path}")
        run_fn = mod.run

        sig = inspect.signature(run_fn)
        params = list(sig.parameters.keys())

        # آماده‌سازی cfg دیکشنری در صورت نیاز
        cfg_dict_to_use = None
        try:
            if isinstance(cfg, dict) and len(cfg) > 0:
                cfg_dict_to_use = cfg
            else:
                # از فایل config (پیش‌فرض یا موقت) بخوان
                if cfg_path_to_use and Path(cfg_path_to_use).exists():
                    cfg_dict_to_use = _load_yaml(Path(cfg_path_to_use))
        except Exception:
            cfg_dict_to_use = None

        # تلاش هوشمندانه برای فراخوانی با امضاهای مختلف
        called = False
        try_orders = []

        if "cfg_path" in params and "out_html" in params:
            try_orders.append(lambda: run_fn(cfg_path=str(cfg_path_to_use), out_html=out_html))
        if "cfg" in params and "out_html" in params and cfg_dict_to_use is not None:
            try_orders.append(lambda: run_fn(cfg=cfg_dict_to_use, out_html=out_html))
        if "cfg_path" in params and "out_html" not in params:
            try_orders.append(lambda: run_fn(cfg_path=str(cfg_path_to_use)))
        if "cfg" in params and "out_html" not in params and cfg_dict_to_use is not None:
            try_orders.append(lambda: run_fn(cfg=cfg_dict_to_use))

        # اگر امضای نام‌دار نبود، شانس با positional بدهیم
        if not try_orders:
            # سعی 1: (cfg_path, out_html)
            try_orders.append(lambda: run_fn(str(cfg_path_to_use), out_html))
            # سعی 2: (cfg, out_html)
            if cfg_dict_to_use is not None:
                try_orders.append(lambda: run_fn(cfg_dict_to_use, out_html))
            # سعی 3: (cfg_path,)
            try_orders.append(lambda: run_fn(str(cfg_path_to_use)))
            # سعی 4: (cfg,)
            if cfg_dict_to_use is not None:
                try_orders.append(lambda: run_fn(cfg_dict_to_use))

        last_err = None
        for attempt in try_orders:
            try:
                attempt()
                called = True
                break
            except TypeError as te:
                last_err = te
                continue

        if not called:
            # اگر هیچ‌کدام جواب نداد، پیام شفاف بده
            raise TypeError(
                f"Cannot call run() with supported signatures. Parameters found: {params}. "
                f"Last error: {last_err}"
            )

    if temp_cfg_path and temp_cfg_path.exists():
        try:
            temp_cfg_path.unlink(missing_ok=True)
        except Exception:
            pass

    return out_html
