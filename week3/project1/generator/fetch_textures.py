"""Download CC0 colour maps (belt + tile surfaces) from Poly Haven / ambientCG into textures/<category>/.
Usage: python fetch_textures.py            -> downloads everything in LIST, writes textures/SOURCES.md
"""
import io, json, os, sys, urllib.request, zipfile
import cv2, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "textures")
UA = {"User-Agent": "Mozilla/5.0 (course-dataset-builder)"}

# (category, source, asset id, output size)
# Culled after visual review in mosaic_v2 (flat, mossy, lacquered, duplicate, mud-crack lines):
DROP = ["Rubber001", "Rubber002", "Rubber003"]   # speckled playground granulate, not belt-like

LIST = [
    # conveyor belt surfaces (rubber / PVC); used as luminance + fine structure, re-tinted per belt type
    ("belt", "acg", "Rubber001", 1024), ("belt", "acg", "Rubber002", 1024), ("belt", "acg", "Rubber003", 1024),
    ("belt", "acg", "Rubber004", 1024), ("belt", "ph", "rubberized_track", 1024), ("belt", "acg", "Fabric030", 1024),
    # tile surfaces at 1024 px (a tile fills up to 85 % of the frame)
    ("terracotta", "acg", "Clay001", 1024), ("terracotta", "acg", "Clay002", 1024), ("terracotta", "ph", "clay_plaster", 1024),
    ("stone", "acg", "Granite001A", 1024), ("stone", "acg", "Granite005A", 1024), ("stone", "acg", "Travertine003", 1024),
    ("marble", "acg", "Marble006", 1024), ("marble", "acg", "Marble012", 1024), ("marble", "acg", "Marble016", 1024),
    ("marble", "acg", "Marble021", 1024),
    ("plain", "acg", "Plaster001", 1024), ("plain", "acg", "Plaster003", 1024), ("plain", "acg", "Concrete019", 1024),
    ("plain", "acg", "Terrazzo005", 1024), ("plain", "acg", "Terrazzo009", 1024),
]


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        return r.read()


def fetch(src, aid):
    """Return (jpeg bytes of colour map, asset page url, download url)."""
    if src == "ph":
        files = json.loads(get(f"https://api.polyhaven.com/files/{aid}"))
        url = files["Diffuse"]["1k"]["jpg"]["url"]
        return get(url), f"https://polyhaven.com/a/{aid}", url
    meta = json.loads(get(f"https://ambientcg.com/api/v2/full_json?type=Material&id={aid}&include=downloadData"))
    dls = meta["foundAssets"][0]["downloadFolders"]["default"]["downloadFiletypeCategories"]["zip"]["downloads"]
    url = [d for d in dls if d["attribute"] == "1K-JPG"][0]["downloadLink"]
    z = zipfile.ZipFile(io.BytesIO(get(url)))
    name = [n for n in z.namelist() if n.endswith("_Color.jpg")][0]
    return z.read(name), f"https://ambientcg.com/view?id={aid}", url


def main():
    rows = []
    only = set(sys.argv[1:])
    for cat, src, aid, size in LIST:
        if (only and aid not in only) or (not only and aid in DROP):
            continue
        out = os.path.join(TEX, cat, aid.lower() + ".jpg")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        try:
            data, page, url = fetch(src, aid)
        except Exception as e:  # noqa
            print("FAIL", aid, e)
            continue
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(out, img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        rows.append((cat, os.path.relpath(out, TEX), aid, "Poly Haven" if src == "ph" else "ambientCG", page, url))
        print("ok", out, os.path.getsize(out) // 1024, "KB")
    with open(os.path.join(HERE, "_dl", "sources.json"), "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
