
"""
Run this file to launch the Streamlit web UI in your browser.
Usage:
    python run_webui.py
"""
import subprocess, sys, shutil
from pathlib import Path

def main():
    this_dir = Path(__file__).parent
    app = this_dir / "webui" / "app.py"
    if not app.exists():
        print("webui/app.py not found next to this runner.", file=sys.stderr)
        sys.exit(1)
    streamlit = shutil.which("streamlit")
    if not streamlit:
        print("Could not find 'streamlit' in PATH. Install with: pip install streamlit", file=sys.stderr)
        sys.exit(1)
    cmd = [streamlit, "run", str(app)]
    print("Launching Streamlit…")
    sys.exit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
