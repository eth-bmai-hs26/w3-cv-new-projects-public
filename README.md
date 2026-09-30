# Warehouse Inspector: automated tile inspection

A computer-vision project for the ETH BMAI course. A recycling warehouse receives reclaimed floor tiles by the
thousand; you build the model that inspects each one. A camera photographs every tile on a conveyor, a U-Net
marks the usable surface, and the station decides whether it can be sold. Then you run your model in the
warehouse's inspection platform on your own laptop.

**Everything is in [`week3/project1/`](week3/project1/)**, and its README explains how to start.

| | |
|---|---|
| 📓 **The notebook** | [open `week3/project1/notebook.ipynb` in Colab](https://colab.research.google.com/github/eth-bmai-hs26/we-cv-new-projects-public/blob/main/week3/project1/notebook.ipynb) (choose a GPU runtime) |
| 🏭 **The platform** | Windows: double-click `week3/project1/platform/start-windows.bat` · macOS: `start-mac.command` · Linux: `./run.sh` |
| 🖼️ **The slides** | [`week3/project1/slides/warehouse_inspector_slides.pdf`](week3/project1/slides/warehouse_inspector_slides.pdf) |

The tile photos are synthetic, made with CC0 textures (see `week3/project1/generator/textures/SOURCES.md`).
