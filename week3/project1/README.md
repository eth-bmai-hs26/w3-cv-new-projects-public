# ♻️ Warehouse Inspector: automated tile inspection

A recycling warehouse receives reclaimed floor tiles by the thousand. Today a person checks every tile by hand. In this project each tile instead rides a conveyor through an **inspection station**. A camera photographs it, and a computer-vision model finds the damaged areas and decides whether the tile can be sold.

Students build that model in a Colab notebook. Then they run it in the **inspection platform**, the web app the warehouse uses every day, on their own laptop.

| | |
|---|---|
| **Students build** | a 3-class U-Net segmenter, three decision heads on its latent space (FNN, CNN, usable-share regressor), a receptive-field experiment, an improved U-Net, and a pretrained ResNet-18 U-Net |
| **Where** | Google Colab (GPU; Colab Pro recommended) for the notebook, their own laptop for the platform |
| **Time** | ~45–60 min of GPU time for a full run of the notebook; the exercise itself takes a few sessions |
| **Stack** | PyTorch · torchvision · self-contained HTML widgets in the notebook · FastAPI + SQLite + React/TypeScript for the platform |
| **Slides** | [`slides/warehouse_inspector_slides.pdf`](slides/warehouse_inspector_slides.pdf): the project, the pipeline and the platform in 22 slides |

---

## Contents

1. [The task](#the-task)
2. [What is in this folder](#what-is-in-this-folder)
3. [Quick start for students](#quick-start-for-students)
   - [A. The notebook in Colab](#a-the-notebook-in-colab)
   - [B. The inspection platform on your laptop](#b-the-inspection-platform-on-your-laptop)
4. [The notebook, section by section](#the-notebook-section-by-section)
5. [Metrics and targets](#metrics-and-targets)
6. [Saving your work and resuming later](#saving-your-work-and-resuming-later)
7. [The dataset](#the-dataset)
8. [The inspection platform](#the-inspection-platform)
9. [Troubleshooting](#troubleshooting)

---

## The task

Each photo shows **one tile on the conveyor belt**. Every pixel belongs to one of three classes:

| Class | Meaning |
|---|---|
| 0 **belt** | background: not part of the tile |
| 1 **usable** | tile surface that can be sold |
| 2 **damaged zone** | cracks, chips and missing pieces, plus a **cutting margin** of 3 % of the tile's short side around each defect (you cannot cut right up to a crack) |

Stains, dirt and scratches are **not** damage. The tile can be cleaned, so they count as usable. Telling a crack from a stain is the hard part of the task.

**The decision rule:**

```
usable share = usable pixels / (usable + damaged pixels)      (measured on the tile, not the whole photo)

usable share ≥ 85 %   →  APPROVE   (sell it, cut down if needed)
usable share < 85 %   →  REJECT    (recycle it)
```

The threshold lives in one place: `data.USABLE_THRESHOLD = 0.85`. The platform adds a **review band** around it: tiles between 80 % and 90 % go to a person.

---

## What is in this folder

```
week3/project1/
├── notebook.ipynb              # 👩‍🎓 the notebook you work in
├── data.py                     # dataset: class maps, usable share, in-memory cache, splits, augmentation
├── utils.py                    # LatentUNetBase, DepthUNet, trainers, evaluation, receptive-field tools,
│                               #   checkpoints (save_model/load_model), export_for_platform
├── widgets.py                  # the interactive HTML panels (dashboards, scoreboard, inspector, …)
├── app_colab.py                # the "you vs. the model" game inside Colab (§7b, §9.7)
├── warehouse_app_colab.html    #   … its page
├── dataset_new/                # ⭐ THE dataset: 5 000 single-tile photos + masks (used everywhere)
├── generator/                  # the procedural generator that made dataset_new (+ CC0 textures)
├── platform/                   # 🏭 the inspection platform (FastAPI + React), runs on a laptop
└── slides/                     # the project presentation (LaTeX source + PDF)
```

---

## Quick start for students

### A. The notebook in Colab

1. **Open the notebook in Colab:**
   `https://colab.research.google.com/github/eth-bmai-hs26/w3-cv-new-projects-public/blob/main/week3/project1/notebook.ipynb`
   (or *File → Open notebook → GitHub*, search for `eth-bmai-hs26/w3-cv-new-projects-public` and pick `week3/project1/notebook.ipynb`).
2. **Choose a GPU.** *Runtime → Change runtime type → GPU*. With Colab Pro pick **L4** or **A100** and turn on **High-RAM**.
3. **Run the cells from the top.** The first cell clones the project and the second shows the inspection line you are building. §1 asks where to save your work: tick **Save to Drive** so your models survive a runtime restart.
4. **Work through the 11 tasks.** Each one has a **🎯 Task N** heading, and every line you fill in is marked `# 🎯 TODO`; the table *Your tasks at a glance* at the top lists them all. Every section ends with a live panel that tells you how well you did.
5. **§11 exports your best model** as `platform_model.pt` and downloads it. You will load that file into the platform (part B).

Cells that only do plumbing (cloning, imports, drawing the panels, downloads) are collapsed into Colab **forms**. Press ▶ to run one; click *Show code* if you are curious. Everything you need to understand or write stays visible.

### B. The inspection platform on your laptop

The platform runs **on your own computer**, not in Colab. You need **Python 3.10 or newer** ([python.org](https://www.python.org/downloads/)) and an internet connection the first time. Node.js is **not** needed, because the web interface ships pre-built.

Get the project folder onto your laptop: `git clone https://github.com/eth-bmai-hs26/w3-cv-new-projects-public.git`, or download it as a ZIP from GitHub (*Code → Download ZIP*) and unzip it. Then:

| System | Start it |
|---|---|
| **Windows** | double-click **`week3\project1\platform\start-windows.bat`**. Or, in PowerShell: `cd week3\project1\platform` then `.\run.ps1` |
| **macOS** | double-click **`week3/project1/platform/start-mac.command`**. The first time, macOS may refuse to open it: right-click it → *Open*. Or, in Terminal: `cd week3/project1/platform && ./run.sh` |
| **Linux** | `cd week3/project1/platform && ./run.sh` |

The first start creates a private Python environment (`platform/.venv`) and installs the packages. That takes a few minutes and around 1 GB of disk, mostly PyTorch. Nothing is installed system-wide. Later starts take seconds.

Your browser then opens at **http://127.0.0.1:8000**. If that port is taken, the next free one is used and printed in the window. Stop the platform with **Ctrl + C** in that window, or close the window.

The platform starts with the course staff's ResNet-18 U-Net and 30 days of demo history. To plug in **your own model**, do one of these:

- **Settings → Model → Upload your model .pt** and pick `platform_model.pt`. It loads at once.
- or start the platform with your model:
  - Windows: `.\run.ps1 --model $HOME\Downloads\platform_model.pt`
  - macOS / Linux: `./run.sh --model ~/Downloads/platform_model.pt`

**Use it from your phone:** start it with `--host 0.0.0.0` (e.g. `./run.sh --host 0.0.0.0`) and open `http://<your-laptop's-IP>:8000` on a phone on the same Wi-Fi. The **Inspect** page then offers the phone camera.

More in [`platform/README.md`](platform/README.md): every page, the API, the decision rule, model formats and tests.

---

## The notebook, section by section

Each section answers one question and hands its answer to the next. The notebook marks these links with 🧭 **Why this step?** notes.

| § | Section | What students do | Why |
|---|---|---|---|
| 1 | 🗂️ Setup | pick a GPU and a storage place; meet Dice loss and **the metrics** (§1.2) | know what "good" will mean before training anything |
| 2 | 🔍 Dataset | 🎯 write a `Dataset` that returns (photo, class map, label); check it against the reference; cache 5 000 tiles in memory; make train/val/test splits (3 500 / 750 / 750) | a model is only as good as the data pipeline feeding it |
| 3 | 🧱 First U-Net | 🎯 build a small U-Net (`encode` / `decode` on `LatentUNetBase`) and 🎯 its training loop | segmentation plus the 85 % rule is already a complete inspector |
| 4 | 🔢 FNN head | 🎯 classify APPROVE/REJECT from the **pooled** bottleneck vector | can the latent space alone decide? |
| 5 | 🧠 CNN head | 🎯 classify from the **spatial** bottleneck map | *where* the damage is matters, and pooling threw it away |
| 6 | ⚖️ Evaluation | scoreboard against targets, clickable confusion matrix, error inspector | find *which* tiles fail and why |
| 7 | 🚀 One tile end to end | follow one photo through the pipeline card; play **you vs. the model** (§7b) | see what the numbers mean on a single tile |
| 8 | 🔭 Receptive field | measure theoretical and effective receptive fields; train `DepthUNet` with depth 2/3/4 | the basic model confuses cracks with stains because it sees too little context |
| 9 | 🏆 Challenge | 🎯 `ImprovedUNet`, 🎯 improved FNN and CNN heads, 🎯 a new **usable-share regressor**; final scoreboard; **business cost** of false approvals vs. false rejections | beat every target and choose a threshold that makes business sense |
| 10 | 🎁 Bonus: transfer learning | 🎯 the decoder of a U-Net on an **ImageNet-pretrained ResNet-18** encoder (frozen, then fine-tuned) | the best model with the least training |
| 11 | 🏭 Deploy | export the best model (TorchScript) and download it | from experiment to product |
| 12 | 📌 Takeaways | the journey from 86 % to 98.5 %, and what each step taught | |

🎯 = one of the 11 tasks (TODO cells).

**Interactive panels** (all in `widgets.py`) are plain HTML/SVG/JS with the data embedded. They work in Colab (including Safari), JupyterLab and VS Code, and they stay visible in a saved notebook:

- live training dashboards;
- the scoreboard against the targets;
- a test-set inspector with error overlays;
- a clickable confusion matrix;
- a threshold and business-cost explorer;
- a receptive-field visualiser (click a spot on a tile to see what each model sees there);
- an architecture diagram;
- side-by-side galleries of the models on the same tiles;
- the animated inspection line.

---

## Metrics and targets

| Metric | What it measures | Target (§9) |
|---|---|---|
| **Accuracy** | share of test tiles with the right APPROVE/REJECT verdict | Threshold ≥ 96 %, FNN ≥ 78 %, CNN ≥ 92 %, Regression ≥ 90 % |
| **IoU** (per class) | overlap / union of the predicted and true pixels of one class | |
| **mIoU** | mean IoU over belt, usable and damaged | ≥ 0.90 |
| **Usable-share MAE** | mean \|predicted − true usable share\| over test tiles, in percentage points | ≤ 2.5 pp |
| **Dice** | 2·overlap / (predicted + true pixels); also the segmentation loss | |

*Threshold* means "segment, then apply the 85 % rule to the predicted mask". *FNN*, *CNN* and *Regression* are the three heads on the U-Net's bottleneck.

**Staff results from a full solution run** (test set of 750 tiles; the targets were set just below these):

| Model | Threshold | FNN | CNN | Regression | mIoU | MAE |
|---|---|---|---|---|---|---|
| Basic U-Net (§3–5) | 86.3 % | 64.8 % | 70.8 % | – | 0.587 | 7.5 pp |
| DepthUNet 2 / 3 / 4 (§8) | 92.4 / 92.5 / 92.8 % | | | | | |
| Improved U-Net (§9) | 97.3 % | 79.5 % | 93.1 % | 91.7 % | 0.935 | 1.86 pp |
| ResNet-18 U-Net (§10) | **98.5 %** | | | 94.1 % | **0.958** | **0.92 pp** |

---

## Saving your work and resuming later

Every model is saved when it finishes training:
- in Colab with **Save to Drive** ticked: to `MyDrive/warehouse-inspector/`, so it survives a runtime restart;
- otherwise: to the session's local `cache/`.

**Coming back days later:**
1. Run the cells marked **🔁** in Sections 1–2 (about two minutes: clone, imports, storage, data).
2. Jump to the section you want to continue and run its **▶ RESUME HERE** cell. Its title says which saved models it reloads. For example, §4 and §5 reload only the §3 U-Net, because they train their own head on top of it.

Checkpoints are whole-model files written with `dill`, so the student's own classes travel inside them. They only load on the Python version that wrote them. That is why §11 exports a **TorchScript** file for the platform instead: it loads on any computer with PyTorch.

---

## The dataset

`dataset_new/` holds **5 000 photos** (512 × 512) of one reclaimed tile on a conveyor belt, with pixel-exact masks:

| Path | Content |
|---|---|
| `original/tile_0001.jpg` | the photo |
| `tiles/tile_0001-tiles.png` | every visible tile pixel |
| `damage/tile_0001-damage.png` | the raw defects (crack lines, chips, missing pieces) |
| `segmented/tile_0001-segmented.png` | the usable area: tile minus the defects widened by the cutting margin |
| `ground_truth.csv` | per tile: usable_fraction, has_damage, damage_types, material, belt, seed, status |

`data.py` turns the masks into the 3-class map: belt = outside `tiles`, usable = `segmented`, damaged = `tiles` and not `segmented`. The notebook trains at 256 × 256, so it recomputes each tile's usable share and label at that size. A handful of borderline labels flip, and the notebook shows how many.

**Mix:**
- 54.6 % APPROVE, 37 % pristine tiles;
- six materials: terracotta, encaustic, glazed, marble, concrete, stone;
- five belt types;
- damage from cracks, chips and shattered pieces;
- harmless stains, dirt and scratches as distractors.

The images are **synthetic**. They come from [`generator/`](generator/), which renders CC0 photo textures (Poly Haven, ambientCG; see `generator/textures/SOURCES.md`) with light-box lighting and camera effects. Generation is deterministic:

```bash
cd week3/project1/generator
pip install numpy opencv-python-headless
python generator.py --dataset ../dataset_new --n 5000 --workers 8 --seed 0     # the course dataset (~15 min)
python generator.py --n 12 --out samples/                                      # a few samples to look at
```

---

## The inspection platform

The software the warehouse runs: operators upload tile photos or take them with a phone. Every tile gets:
- a class map;
- a usable share;
- a verdict with a confidence.

Borderline tiles go to a **review queue**, where a person decides. Managers follow approval rates, recovered value and supplier quality on a dashboard.

| Page | What it is for |
|---|---|
| **Dashboard** | tiles inspected, approval rate, review queue, recovered value (€), throughput, usable-share histogram, quality by supplier |
| **Inspect** | assign photos to a lot, drop in a batch or use the phone camera, watch the tiles move through the camera gate, compare photo and overlay with a slider |
| **Lots** | per-shipment summary, accept or reject a lot, CSV export, printable report |
| **Review** | borderline and low-confidence tiles, least confident first; every decision is signed and lands in the retraining export |
| **Settings** | threshold, review band and minimum confidence, with a live what-if on the last 30 days; prices and costs; the model file |

**How it fits together:**

```
 browser (React + TypeScript, pre-built)  ── /api (JSON) ──▶  FastAPI (one Python process)
                                          ◀─ /media (images) ──  ├── model_service.py  loads the .pt, runs the U-Net
                                                                ├── rules.py          THE decision rule + money
                                                                ├── imaging.py        overlays, photo storage (HEIC too)
                                                                └── db.py             SQLite in platform/var/
```

The platform preprocesses a photo exactly as the notebook does: greyscale, 256 × 256, values in [0, 1]. It then takes the argmax of the model's 3-class logits and computes the usable share. `rules.py` decides:

| Usable share | Verdict |
|---|---|
| ≥ 90 % | APPROVE |
| 80 % up to 90 % | REVIEW (a person decides) |
| < 80 % | REJECT |

A tile also goes to REVIEW when the model is unsure of its own mask (confidence below 40 %). All of these numbers can be changed under Settings.

Without a model file the platform runs in a clearly labelled **DEMO MODE** (a colour/texture heuristic). The shipped `platform/models/default_model.pt` is the staff ResNet-18 U-Net (98.5 % test accuracy).

Full documentation: [`platform/README.md`](platform/README.md).

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError` in the first cell after deleting the cloned folder | *Runtime → Restart session*, then run the first cell again (it clones afresh). |
| Models gone after a restart | "Save to Drive" was unticked, so they were in the session's local storage. Tick it and retrain; the next restart keeps them. |
| `load_model` fails with "no locals found when setting up annotations" (or similar) | A `save_model` checkpoint only loads on the Python version that wrote it. Load it in Colab, or use the TorchScript file from §11. |
| Interactive panel blank or red | Re-run the cell. A red box shows the JavaScript error: send it to the course staff. |
| Windows: "running scripts is disabled on this system" | Use `start-windows.bat`: it runs `run.ps1` without changing the policy. Or run `powershell -ExecutionPolicy Bypass -File .\run.ps1`. |
| Windows: `python` opens the Microsoft Store | Install Python from python.org and tick **Add python.exe to PATH**. The launcher tries the `py` launcher first. |
| macOS: "cannot be opened because it is from an unidentified developer" | Right-click `start-mac.command` → *Open* → *Open*. Or run `./run.sh` in Terminal. |
| `permission denied: ./run.sh` | `chmod +x run.sh start-mac.command` (a ZIP download can lose the executable bit). |
| Platform install stopped halfway (network) | Start it again. It notices the unfinished install and repeats it. To start clean, delete `platform/.venv`. |
| Platform shows **DEMO MODE** | `platform/models/default_model.pt` is missing (e.g. a partial download). Re-download it, or upload your own model under Settings. |
| iPhone photos (HEIC) are rejected | `pillow-heif` could not be installed. In `platform/`: `.venv/bin/python -m pip install pillow-heif` (Windows: `.venv\Scripts\python -m pip install pillow-heif`), or export the photos as JPEG. |
| The phone cannot reach the platform | Start it with `--host 0.0.0.0`, use the laptop's IP on the same Wi-Fi, and allow Python through the firewall when the system asks. |
