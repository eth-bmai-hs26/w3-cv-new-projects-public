"""
Launch the Warehouse Inspector app inside a Colab notebook.

The app is a self-contained HTML page (warehouse_app_colab.html) that talks
to Python through Colab callbacks:

    getTiles()                 -> {"threshold": …, "tiles": test tiles with true and model verdicts}
    getImage(filename, kind)   -> base64 image; kind = original | truth | prediction | errors

Everything is served from memory (the TileCache and an EvalResult), so running
the app never touches the dataset on disk.
"""

import base64
import io
import os

import numpy as np
from IPython.display import HTML, JSON, display
from PIL import Image

from week3.project1.data import USABLE_THRESHOLD
from week3.project1.widgets import _resize_cls, class_rgb, error_overlay, test_inspector

APP_HTML = os.path.join(os.path.dirname(__file__), "warehouse_app_colab.html")


def _png(arr, size):
    im = Image.fromarray(arr)
    if im.size[0] != size:
        im = im.resize((size, size), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode(), "image/png"


def _jpeg(arr, size):
    im = Image.fromarray(arr).convert("RGB")
    if im.size[0] != size:
        im = im.resize((size, size), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"


def launch_warehouse_app(res, image_size=384):
    """
    Show the Warehouse Inspector for the test tiles in `res` (an EvalResult).
    Challenge mode: you vs. the model, both scored against the TRUE label.
    Explorer mode:  photo vs. model mask, with true and predicted usable share.
    """
    try:
        from google.colab import output
    except ImportError:
        print("The Warehouse app needs Google Colab (it uses Colab callbacks). "
              "Showing the notebook inspector instead.")
        test_inspector(res)
        return

    df, cache = res.df, res.cache
    row_of = {f: r for r, f in enumerate(df["filename"])}
    tiles = [{
        "index": int(r),
        "filename": df["filename"].iat[r],
        "true_usable": float(df["true_frac"].iat[r]),
        "true_status": df["Actual"].iat[r],
        "model_usable": float(df["pred_frac"].iat[r]),
        "model_status": "APPROVE" if df["pred_frac"].iat[r] >= USABLE_THRESHOLD else "REJECT",
    } for r in range(len(df))]

    def get_tiles():
        return JSON({"threshold": USABLE_THRESHOLD, "tiles": tiles})

    def get_image(filename, kind):
        r = row_of[filename]
        idx = int(df["index"].iat[r])
        pred = _resize_cls(res.pred_masks[r], cache.size)
        if kind == "original":
            data, mime = _jpeg(cache.image(idx), image_size)
        elif kind == "truth":
            data, mime = _png(class_rgb(cache.target(idx)), image_size)
        elif kind == "errors":
            data, mime = _jpeg(error_overlay(cache.image(idx), cache.target(idx), pred), image_size)
        else:  # "prediction" (and the legacy name "segmented")
            data, mime = _png(class_rgb(pred), image_size)
        return JSON({"image": data, "mime": mime})

    output.register_callback("getTiles", get_tiles)
    output.register_callback("getImage", get_image)
    with open(APP_HTML) as f:
        display(HTML(f.read()))
