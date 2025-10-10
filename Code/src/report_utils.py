# Code/src/report_utils.py
from __future__ import annotations
import base64, io, os, datetime
from typing import Tuple
from pathlib import Path

def ensure_dir(path: str | os.PathLike) -> str:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)

def _png_datauri_from_fig(fig, dpi: int = 140) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"

def save_fig_dual(fig, out_dir: str, name: str, dpi: int = 140) -> Tuple[str, str]:
    """
    Save figure as PNG in out_dir and also return a data: URI.
    Returns (data_uri, file_path).
    """
    ensure_dir(out_dir)
    fn = f"{name}.png"
    fpath = os.path.join(out_dir, fn)
    fig.savefig(fpath, format="png", dpi=dpi, bbox_inches="tight")
    data_uri = _png_datauri_from_fig(fig, dpi=dpi)
    return data_uri, fpath

def timestamped_assets_dir(base_dir: str = "report_assets") -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_dir(os.path.join(base_dir, ts))
