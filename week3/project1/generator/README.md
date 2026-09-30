# Tile dataset generator

Procedural generator for the project's dataset: photos of **one reclaimed tile on the conveyor belt of an inspection station**, with pixel-exact masks. It renders varied belts, CC0 photo-textured materials (marble, granite, terrazzo, terracotta, glazed and patterned cement tiles), realistic damage (cracks, chips, shattered pieces) and harmless distractors (stains, dirt, scratches), then applies light-box lighting and camera effects (slight perspective, blur, noise, JPEG).

```bash
pip install numpy opencv-python-headless
python generator.py --dataset ../dataset_new --n 5000 --workers 8 --seed 0   # the course dataset (~15 min)
python generator.py --n 12 --out samples/                                    # a few samples to look at
```

Output layout (all masks are 0/255 PNGs aligned with the photo):

| Path | Content |
|---|---|
| `original/tile_0001.jpg` | the photo |
| `tiles/tile_0001-tiles.png` | every visible tile pixel |
| `damage/tile_0001-damage.png` | raw defect pixels (crack lines, chip areas, missing pieces) |
| `segmented/tile_0001-segmented.png` | usable area = tile minus the defects dilated by a cutting margin (3 % of the tile's short side) |
| `ground_truth.csv` | per tile: usable_fraction = usable / tile, has_damage, damage_types, material, belt, seed |

The notebook derives the three classes from these masks: background (outside `tiles`), usable (`segmented`) and damaged (`tiles` and not `segmented`). A tile is approved when `usable_fraction >= 0.85`.

Generation is deterministic for a given `--seed`. `textures/SOURCES.md` lists every texture with its source; all are **CC0 1.0** (Poly Haven, ambientCG). `fetch_textures.py` re-downloads them; without `textures/` the generator falls back to procedural materials. `tools.py` makes preview sheets and mask-alignment overlays.
