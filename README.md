# Do networks use the order of events? SHD and SSC

An independent research project that extends my UCLA COSMOS 2026 project on the energy efficiency of
spiking neural networks (the rest of this repository).

Spiking networks are often said to use the timing of events. These Colab notebooks test whether
networks trained on two spoken-word datasets, SHD and SSC, actually use the order of time windows, and
what each model would cost in energy.

Every recording is cut into time windows of spike counts (one row per window, one column per input
channel). Three kinds of model read these tables:

- a **count classifier** sees only each channel's total over the recording;
- **order-blind designs** see every window but never their order; the best of them on held-out
  recordings is the order-blind ceiling *r*;
- **order-using networks** (spiking networks, a recurrent network, LSTMs and RadLIF) are trained twice
  from the same seed: windows in order (test accuracy *a0*) and windows shuffled (*a1*).

```
drop = a0 - a1 = (a0 - r) + (r - a1) = gain + deficit
```

A network can only be said to need the order by at most its gain over the order-blind ceiling. Energy
per recording is computed (not measured) with a hardware-aware accounting, in three cases from no data
reuse to weights kept on the chip.

The notebooks are written for readers who know how a neural network learns but are new to code: every
step has a short explanation, and the comments on the lines of code say what each line does.

## Run it in Colab

| notebook | what it does | GPU | what Run all does as the notebook is set |
|---|---|---|---|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Avi-Seth/UCLA-COSMOS-SNN-Energy-efficiency/blob/main/SHD-SSC-order-of-events/01_explore_datasets.ipynb) `01_explore_datasets` | Opens SHD and SSC (and, if switched on, DVS128 Gesture), prints and draws recordings, cuts them into time windows exactly as the training notebooks do, and shows what the experiments do to the windows (shuffling, reversing, joining). | no | downloads SHD and SSC and draws everything |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Avi-Seth/UCLA-COSMOS-SNN-Energy-efficiency/blob/main/SHD-SSC-order-of-events/02_train_SHD.ipynb) `02_train_SHD` | SHD: defines every model, with why it is used, its architecture and how its energy is computed; runs self-checks; trains every model; saves accuracy and energy for every run. | yes | a quick test of the whole notebook |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Avi-Seth/UCLA-COSMOS-SNN-Energy-efficiency/blob/main/SHD-SSC-order-of-events/03_train_SSC.ipynb) `03_train_SSC` | The same for SSC. | yes | a quick test of the whole notebook |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Avi-Seth/UCLA-COSMOS-SNN-Energy-efficiency/blob/main/SHD-SSC-order-of-events/04_analyze_results.ipynb) `04_analyze_results` | Chooses the ceilings on held-out recordings, computes drop, gain and deficit with the decision rule, and makes every table (CSV and Markdown) and figure (PNG and PDF). | no | every table and figure of the study, from `results.zip`, in about a minute |

1. Click a badge. The notebook opens in Colab.
2. For `02` and `03`, choose a GPU: Runtime > Change runtime type.
3. Runtime > Run all.

Nothing else is needed. To keep your own changes to a notebook, use File > Save a copy in Drive.

**The data.** SHD and SSC are downloaded from the dataset authors' site, https://zenkelab.org/datasets/
(SHD 169 MB; SSC 1.7 GB for all three of its files, 1.5 GB for the two that `03` uses). Every download
is checked against the checksum the site publishes, then unpacked on the Colab machine. If you already
have the files, for example on your Google Drive, set `MY_DATA_DIR` in Step 1 of the notebook to their
folder and nothing is downloaded. DVS128 Gesture, used only in `01`, is off by default; `RUN_DVS = True`
downloads it with the tonic library.

**Where the outputs go.** A Colab machine and everything on it is deleted when the session ends. So by
default each notebook's last cell packs its outputs into one `.zip` file and offers it as a download.
With `USE_DRIVE = True` (Step 1) the outputs are saved on your Google Drive instead, in the folder
`shd_ssc_order/`; Colab asks once for permission.

## The full study

`02` and `03` run a quick test as they are set: a few recordings and one or two training epochs, so the
numbers mean nothing, but every cell runs. For the full study, set `QUICK = False` in Step 1. The
notebook then saves every run on your Google Drive the moment it ends. A full run can last longer than
a Colab session; if the session stops, open the notebook again and choose Run all: runs already saved
are skipped. For `03` also choose High-RAM (Runtime > Change runtime type): its 75,466 training
recordings take about 5 GB of memory as 8-bit counts.

Planning figures for an NVIDIA L4 GPU, on which the published results were computed. Before training,
every notebook times each model on the GPU you have and prints its own estimate; a progress bar then
shows the time left.

| notebook | experiment | what | runs | GPU hours (L4) |
|---|---|---|---|---|
| `02_train_SHD` | `shd_controls` | control tasks with known answers (chains, reversal, negative control) | 150 | about 1.6 |
| `02_train_SHD` | `shd_25` | SHD, 25 windows per recording, five seeds | 125 | under 2 |
| `02_train_SHD` | `shd_14ms` | SHD, 100 windows of 14 ms, five seeds | 105 | under 7 |
| `03_train_SSC` | `ssc_10ms` | SSC, 100 windows of 10 ms, three seeds | 48 | about 15 to 16 |

`02` and `03` do not depend on each other: run them one after the other or, if your Colab account allows
two GPU sessions at once, at the same time.

## The analysis

`04_analyze_results` needs no GPU. As it is set (`RESULTS = 'published'`), it downloads `results.zip`
from this folder of the repository: the saved runs of the study (428 runs, one small file each, with every model's
accuracy, its answer on every test recording and its energy estimate). It then remakes every table and
figure. To analyse your own full runs instead, set `RESULTS = 'mine'`: the notebook reads them from
`MY_RESULTS_DIR` on your Google Drive, and skips whatever has not been run yet.

## Files

| file | what it is |
|---|---|
| `01_explore_datasets.ipynb` to `04_analyze_results.ipynb` | the four notebooks |
| `results.zip` | the saved runs of the study: `results/shd_controls`, `results/shd_25`, `results/shd_14ms` and `results/ssc_10ms`, each with a `meta.json` (the experiment's description) and a `runs/` folder (one `.json` file per run) |
| `REFERENCES.md` | every source the notebooks rely on |
| `requirements.txt` | the Python packages, for running outside Colab |

## Running outside Colab

The notebooks are written for Colab. To run one in Jupyter on your own computer, install the packages
of `requirements.txt` and change the folders of Step 2 (`DATA_DIR`, and `BASE_DIR` or `OUT_DIR`), which
point to Colab's `/content/` folder. Google Drive is not needed when `USE_DRIVE = False`.

## Data, code and credits

- SHD and SSC: B. Cramer, Y. Stradmann, J. Schemmel and F. Zenke, The Heidelberg Spiking Data Sets for
  the Systematic Evaluation of Spiking Neural Networks, IEEE Transactions on Neural Networks and
  Learning Systems 33(7), 2744-2757 (2022), doi:10.1109/TNNLS.2020.3044364. The datasets are licensed
  under Creative Commons BY 4.0 (https://creativecommons.org/licenses/by/4.0/) and are downloaded from
  https://zenkelab.org/datasets/; they are not part of this repository.
- DVS128 Gesture: A. Amir et al., A Low Power, Fully Event-Based Gesture Recognition System, IEEE
  Conference on Computer Vision and Pattern Recognition (CVPR) 2017, 7243-7252; the version used is the
  one the tonic library downloads (https://tonic.readthedocs.io).
- RadLIF: A. Bittar and P. N. Garner, A surrogate gradient spiking baseline for speech command
  recognition, Frontiers in Neuroscience 16, 865897 (2022), doi:10.3389/fnins.2022.865897. The RadLIF
  code in `02_train_SHD.ipynb` is adapted from SpArch (https://github.com/idiap/sparch), Copyright 2022
  Idiap Research Institute, BSD-3-Clause; its notice is kept in the notebook.
- Leaky and recurrent leaky neurons as in snnTorch: J. K. Eshraghian et al., Training Spiking Neural
  Networks Using Lessons From Deep Learning, Proceedings of the IEEE 111(9), 1016-1054 (2023),
  doi:10.1109/JPROC.2023.3308088. The training notebooks check that their neurons give the same
  outputs and gradients as snnTorch 1.0.0's.
- Energy accounting: M. Dampfhoffer, T. Mesquida, A. Valentian and L. Anghel, Are SNNs Really More
  Energy-Efficient Than ANNs? An In-Depth Hardware-Aware Study, IEEE Transactions on Emerging Topics in
  Computational Intelligence 7(3), 731-741, doi:10.1109/TETCI.2022.3214509; costs per operation based
  on M. Horowitz, Computing's energy problem (and what we can do about it), IEEE International
  Solid-State Circuits Conference (ISSCC) 2014, Digest of Technical Papers, 10-14,
  doi:10.1109/ISSCC.2014.6757323.

The full reference list is in `REFERENCES.md`.
