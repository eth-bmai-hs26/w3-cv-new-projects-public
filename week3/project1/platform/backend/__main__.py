"""
Command line (run inside platform/):

    python -m backend                      start the platform on http://127.0.0.1:8000
    python -m backend --model my_unet.pt   ... with a model downloaded from Colab
    python -m backend seed --tiles 360     rebuild the demo history
    python -m backend samples OUT --n 60   write placeholder tile photos
"""

from __future__ import annotations

import argparse
import os
import sys
import webbrowser
from threading import Timer


def _free_port(host, port, tries=20):
    """`port` if it can be bound on `host`, else the next free one (up to `tries` further)."""
    import socket
    for p in range(port, port + tries):
        with socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, p))
                return p
            except OSError:
                continue
    return port                                    # let uvicorn report the error


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m backend", description="Tile inspection platform")
    sub = ap.add_subparsers(dest="cmd")

    def common(p):
        p.add_argument("--data", help="folder for the database and stored photos (default platform/var)")
        p.add_argument("--model", help="model checkpoint (.pt) to load, e.g. one downloaded from Colab")
        p.add_argument("--device", default=None, help="auto | cpu | cuda | mps")

    serve = sub.add_parser("serve", help="start the web platform (default)")
    common(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--no-seed", action="store_true", help="do not create demo history on an empty database")
    serve.add_argument("--open", action="store_true", help="open the browser")
    seed = sub.add_parser("seed", help="(re)build the demo history")
    common(seed)
    seed.add_argument("--tiles", type=int, default=360)
    seed.add_argument("--days", type=int, default=30)
    seed.add_argument("--images", help="folder of tile photos (default: the configured image source)")
    samples = sub.add_parser("samples", help="write placeholder single-tile photos")
    samples.add_argument("out")
    samples.add_argument("--n", type=int, default=60)

    # `python -m backend --model x.pt` means serve
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-"):
        argv = ["serve"] + argv
    a = ap.parse_args(argv)

    if a.cmd == "samples":
        from .sample_tiles import write_folder
        write_folder(a.out, a.n)
        print(f"wrote {a.n} photos to {a.out}/original")
        return

    if a.data:
        os.environ["TILE_PLATFORM_DATA"] = a.data
    from .core import Platform
    model = os.path.abspath(a.model) if a.model else None
    plat = Platform(checkpoint=model, device=a.device)
    info = plat.model.info
    print(f"model: {info['name']} ({info['mode']})" + (f" - {info['warning']}" if info.get("warning") else ""))

    if a.cmd == "seed":
        from .seed import seed as run_seed
        run_seed(plat, n_tiles=a.tiles, days=a.days, image_dir=a.images)
        return

    if not a.no_seed and plat.db.one("SELECT COUNT(*) AS n FROM inspections")["n"] == 0:
        from .seed import seed as run_seed
        print("empty database: creating 30 days of demo history (use --no-seed to skip)")
        run_seed(plat)

    import faulthandler
    import signal
    if hasattr(signal, "SIGUSR1"):                 # `kill -USR1 <pid>` prints every thread's stack (debugging hangs)
        faulthandler.register(signal.SIGUSR1, all_threads=True)

    import uvicorn
    from .app import create_app
    app = create_app(plat)
    port = _free_port(a.host, a.port)
    if port != a.port:
        print(f"port {a.port} is in use (another program, or the platform is already running): using {port}")
        a.port = port
    url =f"http://{'127.0.0.1' if a.host in ('0.0.0.0', '::') else a.host}:{a.port}"
    print(f"\n  Tile inspection platform: {url}\n")
    if a.open:
        Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
