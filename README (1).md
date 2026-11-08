# SNNs: Spiking Neural Networks, and their Advantages

I am a 2026 COSMOS scholar, and I was part of Cluster 1. My project, which has culminated over the course of 4 weeks of discussion, looks at spiking neural networks and the claimed energy efficiency "advantage" that it holds, according to popularized research papers. Claude Opus used in assistance to code specific sampling methods and create LeNet-style CNN.

For the overview of the project and the poster, please see `COSMOS_SNN_Project.png` (Project Poster).

# Project Setup

Note: The Project uses two single-layer neural network, which may lead to biased results.

`ann_vs_lif_normal_sized.ipynb`: Contains Control and Experimental Groups 1\
`ann_vs_lif_embedding_sized.ipynb` Contains Control and Experimental Groups 2

Control Group 1: Single-layer Perceptron, 784\
Experimental Group 1: Single-layer LIF Neuron, Poisson Rate-Encoding, 784\
Control Group 2: Single-layer Perceptron, 84\
Experimental Group 2: Single-layer LIF Neuron, Poisson Rate-Encoding, 84

PoMNIST: Poisson Rate-Encoded MNIST data used for Experimental Group 1\
EmbedMNIST: CNN-converted MNIST data with preserved labels, used for Group class 2; encoded into Poisson spike rates within `ann_vs_lif_embedding_sized.ipynb` for Experimental Group 2

Originally planned to use SHD (Spiking Heidelberg Digits) dataset, SHD data does not allow for equal size comparison with normal 784-input layer.

Visuals within code files (not shown in project poster):
- Customizable Raster plot per sample for neural activity
- Spike graph per sample for neural activity

# Credit

Credit to Prof. H. Tad Blair of the UCLA Psychology Department for helping me with some of the core ideas for experimental construction in this AI project.

# Independent research: do networks use the order of events? (SHD and SSC)

After COSMOS I continued with an independent project on the Spiking Heidelberg Digits (SHD), the dataset
I had first planned to use, and the larger Spiking Speech Commands (SSC). It asks whether networks trained
on these spoken-word spike datasets need the order of time windows, or whether a model that never sees
the order does just as well, and what the accuracy that may need order costs in energy. Everything is in
the folder [`SHD-SSC-order-of-events/`](SHD-SSC-order-of-events/), with its own README:

| notebook | | GPU |
|---|---|---|
| `01_explore_datasets` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Avi-Seth/UCLA-COSMOS-SNN-Energy-efficiency/blob/main/SHD-SSC-order-of-events/01_explore_datasets.ipynb) | no |
| `02_train_SHD` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Avi-Seth/UCLA-COSMOS-SNN-Energy-efficiency/blob/main/SHD-SSC-order-of-events/02_train_SHD.ipynb) | yes |
| `03_train_SSC` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Avi-Seth/UCLA-COSMOS-SNN-Energy-efficiency/blob/main/SHD-SSC-order-of-events/03_train_SSC.ipynb) | yes |
| `04_analyze_results` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Avi-Seth/UCLA-COSMOS-SNN-Energy-efficiency/blob/main/SHD-SSC-order-of-events/04_analyze_results.ipynb) | no |
