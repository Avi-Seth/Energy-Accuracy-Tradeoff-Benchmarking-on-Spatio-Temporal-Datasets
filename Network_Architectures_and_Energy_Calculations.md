# Network Architectures and Energy Accounting


## 1. What this document covers

This document describes every model trained in the study and how its energy per recording is computed. The energy is computed from counts of memory accesses and arithmetic operations, following the hardware-aware approach of Dampfhoffer et al. (2022). It is not measured on a chip. Model names are written in readable form, with the name used in the code in brackets.

## 2. The input every model reads

A recording in SHD (Spiking Heidelberg Digits, 20 classes) or SSC (Spiking Speech Commands, 35 classes) is a list of spikes on 700 input channels. Every recording is cut into time windows, and the spikes of each channel in each window are counted. Every model therefore reads a table with one row per window and 700 columns.

**Table 1.** The three settings.

| Setting | Windows per recording | Models trained |
|---|---|---|
| SHD, 25 windows | 25, spanning each recording | all order-blind designs; SNN-FF, SNN-R, RNN, LSTM; RadLIF |
| SHD, 14 ms windows | 100 windows of 14 ms | all order-blind designs; SNN-R, LSTM, RadLIF |
| SSC, 10 ms windows | 100 windows of 10 ms | order-blind designs of 128 to 512 units; SNN-FF, SNN-R, RNN, LSTM |

## 3. The three kinds of model

- Count classifier: sees only each channel's total number of spikes over the whole recording. It is the floor that every other model should beat.
- Order-blind designs: apply the same small network to every window and combine the results by adding, averaging or taking the maximum, which cannot depend on the order of the windows. The best of them on held-out recordings is the order-blind ceiling.
- Order-using networks: read the windows in order (spiking networks, a plain recurrent network, an LSTM and RadLIF). Each is trained twice from the same seed: with the windows in order, and with every recording's windows shuffled.

Every model is trained with 5 seeds on SHD and 3 seeds on SSC. In most models every channel is first scaled as $(\text{count} - \text{mean}) / \text{spread}$, with numbers fixed from the training recordings.

## 4. The models

**Table 2.** Every model. "OB" means order-blind. Classes: 20 on SHD, 35 on SSC. Same-size pairs: OB scores 128 and SNN-FF 128; OB mean 128, SNN-R 128 and RNN 128; OB mean 512 and LSTM 128; OB 2x1,024 and RadLIF.

| Model (code name) | Architecture | Weights (SHD; SSC) |
|---|---|---|
| Count classifier (count_256) | channel totals over the recording $\to$ 256 (ReLU) $\to$ classes | 184,596; 188,451 |
| OB scores 128 (perminv) | every window $700 \to 128$ (ReLU) $\to$ class scores; the scores of all windows added up | 92,308; 94,243 |
| OB mean 128 (pool_128) | every window $700 \to 128$ (ReLU); average over windows $\to 128$ (ReLU) $\to$ classes | 108,820; 110,755 |
| OB mean 512 (pool_512) | as OB mean 128, with 512 units per window | 427,156; 429,091 |
| OB 2x128 (meanmax_deep_128) | every window $700 \to 128 \to 128$ (ReLU after each); average and maximum over windows (256) $\to 128$ (ReLU) $\to$ classes; trained at input dropout 0, 0.2, 0.5 and 0.7 | 141,716; 143,651 |
| OB 2x1,024 (ob_deep_1024_d10) | as OB 2x128 at 1,024 units, with batch normalisation and dropout 0.1: the order-blind design of RadLIF's size | 3,889,172; not used |
| SNN-FF 128 (lif_std) | 128 leaky integrate-and-fire neurons fed by the 700 channels; a readout $128 \to$ classes at every window, added up over the windows | 92,308; 94,243 |
| SNN-R 128 (rlif) | as SNN-FF 128, plus recurrent weights from the 128 neurons to themselves | 108,820; 110,755 |
| RNN 128 (rnn) | a plain recurrent layer (tanh) $700 \to 128$; readout at every window, added up | 108,820; 110,755 |
| LSTM 128 (lstm_d0 to lstm_d70; lstm_paper) | one LSTM layer of 128 cells. SHD: readout at every window, added up; input dropout chosen on held-out recordings. SSC: the dataset authors' set-up, dropout 0.2 on the input and the recurrent state, answer read at the last window | 427,540; 428,963 |
| RadLIF 2x1,024 (radlif) | two layers of 1,024 adaptive leaky integrate-and-fire neurons with recurrent weights, batch normalisation and dropout 0.1; a leaky non-spiking readout (Bittar and Garner, 2022) | 3,895,356; not used |

### 4.1 The spiking neurons

SNN-FF 128 and SNN-R 128 use leaky integrate-and-fire neurons, written out to match snnTorch 1.0.0 (a self-check confirms that they agree). In every window, in order:

$$
v \leftarrow \beta\, v + W_{\text{in}}\, x \;\bigl[+\, W_{\text{rec}}\, s_{\text{prev}}\bigr] - s_{\text{prev}}
$$

$$
s \leftarrow \begin{cases} 1 & \text{if } v > 1 \\ 0 & \text{otherwise} \end{cases}
$$

$v$ is the neuron's voltage, $x$ the counts of the window, $s$ the spikes; $\beta = 0.9$ per window at 25 windows, and the same memory in milliseconds for finer windows. A neuron that spiked loses 1 from its voltage. Because a spike has no useful slope, training uses a smooth stand-in for it (a "surrogate gradient", the fast sigmoid $1 / (25\,\lvert v \rvert + 1)^2$).

RadLIF neurons add an adaptation variable $w$ that rises after each spike and makes the neuron harder to fire; the constants $\alpha$, $\beta$, $a$ and $b$ are learned for every neuron:

$$
\begin{aligned}
w &\leftarrow \beta\, w + a\, u + b\, s \\
u &\leftarrow \alpha\,(u - s) + (1 - \alpha)\,(\text{input} + \text{recurrent input} - w) \\
s &\leftarrow 1 \ \text{ if } u > 1
\end{aligned}
$$

### 4.2 The LSTM cell

In every window, four gates are computed from the input and the last output. The cell's memory and the output are

$$
c \leftarrow f \odot c + i \odot \tilde{c}, \qquad h \leftarrow o \odot \tanh(c)
$$

where $f$, $i$ and $o$ are the forget, input and output gates, $\tilde{c}$ is the candidate and $\odot$ multiplies element by element. The gates let the LSTM keep or discard information over many windows.

## 5. How energy is computed

### 5.1 The idea, from Dampfhoffer et al. (2022)

Dampfhoffer et al. point out that many comparisons of spiking and conventional networks ignore memory accesses, although they are a major share of the energy, or assume a naive conventional implementation. They count memory accesses and operations for both kinds of network on digital hardware, and find that spiking networks compete with efficient conventional implementations only when spikes are very sparse (about 0.15 to 1.38 spikes per synapse per inference, depending on the conventional implementation). This study follows the same idea: every model is charged on the same imagined digital chip, for every memory access and every arithmetic operation.

### 5.2 The costs per operation

Numbers are 8-bit, and the costs are those of a 45 nm technology from Horowitz (2014). Memory accesses cost the most.

**Table 3.** The three basic costs.

| Symbol | Cost | What it is |
|---|---|---|
| $m$ | 1.25 pJ | read or write one 8-bit number (a 64-bit read from an 8 KB memory, 10 pJ, divided by 8) |
| $c$ | 0.23 pJ | one 8-bit multiply-add (multiplication 0.2 pJ + addition 0.03 pJ) |
| $a$ | 0.03 pJ | one 8-bit addition |

**Table 4.** Costs of the operations the models perform.

| Operation | Cost | pJ | What is charged |
|---|---|---|---|
| spike delivered to one synapse | $3m + a$ | 3.78 | read the weight; read and write the target neuron's state; one addition |
| first-layer multiply-add on a non-empty cell | $3m + c$ | 3.98 | read the weight; read and write the partial sum |
| any other multiply-add, no reuse | $4m + c$ | 5.23 | read input and weight; read and write the partial sum |
| multiply-add with ideal reuse | $m + 4c$ | 2.17 | weight read once; the rest in registers |
| multiply-add, weights kept on chip | $4c + m/T$ | depends on $T$ | weight read once per recording of $T$ windows |
| spike with its weight in a register | $2m + c + a$ | 2.76 |  |
| neuron state update | $2m + c$ | 2.73 | read the state, write it back, one multiply-add |
| spike added to a channel total | $2m + a$ | 2.53 | count classifier |

### 5.3 Three cases of data reuse

- Worst case: no data is reused; every operand is read from memory each time.
- Best case within a window: every weight is read once per window.
- Best case across the windows: the weights stay on the chip for the whole recording.

Models are only compared within one case. A conclusion that holds in all three cases does not depend on how well the hardware reuses data.

### 5.4 The formulas

Per test recording: $T$ windows, $C = 700$ channels, $S$ input spikes, $N$ non-empty (window, channel) cells.

Conventional networks and order-blind designs. $W_1$ is the width of the first layer, $P$ the multiply-adds per window after the first layer, and $Q$ the multiply-adds of a part that runs once per recording. Empty cells cost only their read:

$$
E_{\text{worst}} = T\,C\,m + N\,W_1\,(3m + c) + (T\,P + Q)\,(4m + c)
$$

$$
E_{\text{within}} = T\,C\,m + (N\,W_1 + T\,P + Q)\,(m + 4c)
$$

$$
E_{\text{across}} = T\,C\,m + (N\,W_1 + T\,P)\left(4c + \frac{m}{T}\right) + Q\,(m + 4c)
$$

Spiking networks. The first layer is charged either per input spike, $S\,H\,(3m + a)$, or per non-empty cell, $N\,H\,(3m + c)$, where $H$ is the number of neurons each input reaches. The worst case takes the dearer of the two; the within and across cases take the cheaper of two options:

$$
E_{\text{first layer}} =
\begin{cases}
\max\bigl(S\,H\,(3m + a),\; N\,H\,(3m + c)\bigr) & \text{worst} \\[4pt]
\min\bigl(S\,H\,(3m + a),\; N\,H\,(m + 4c)\bigr) & \text{within} \\[4pt]
\min\bigl(S\,H\,(2m + c + a),\; N\,H\,(4c + m/T)\bigr) & \text{across}
\end{cases}
$$

Every hidden spike costs $3m + a$ per synapse it reaches ($2m + c + a$ across the windows), and every neuron state costs $2m + c$ per window. The hidden spikes are counted on the test recordings, so a network that spikes less is charged less.

### 5.5 What each model is charged

**Table 5.** The numbers put into the formulas for each model.

| Model | Charged as ($K$ = number of classes) |
|---|---|
| Count classifier | each input spike added to its channel total ($2m + a$); the 700 totals read once; its two layers once per recording on the non-empty totals |
| OB scores 128 | $W_1 = 128$; $P = 128 \times K$; $Q = 0$ |
| OB mean $h$ ($h = 128$ or $512$) | $W_1 = h$; $P = 0$; $Q = h \times 128 + 128 \times K$ |
| OB 2x128 | $W_1 = 128$; $P = 128 \times 128$; $Q = 256 \times 128 + 128 \times K$ |
| OB 2x1,024 | $W_1 = 1{,}024$; $P = 1{,}024 \times 1{,}024$; $Q = 2{,}048 \times 1{,}024 + 1{,}024 \times K$ (batch normalisation folds into the weights) |
| RNN 128 | $W_1 = 128$; $P = 128 \times 128 + 128 \times K$; $Q = 0$ |
| LSTM 128, SHD | $W_1 = 4 \times 128$ (four gates); $P = 4 \times 128 \times 128 + 128 \times K$; $Q = 0$ |
| LSTM 128, SSC | $W_1 = 4 \times 128$; $P = 4 \times 128 \times 128$; $Q = 128 \times 35$ (readout once, at the last window) |
| SNN-FF 128 | each input reaches 128 neurons; each hidden spike reaches the $K$ readout units; 128 state updates per window |
| SNN-R 128 | as SNN-FF 128, but each hidden spike also reaches the 128 neurons of its layer |
| RadLIF 2x1,024 | each input reaches 1,024 neurons; a layer-1 spike reaches layer 2 (1,024) and its own layer (1,023); a layer-2 spike reaches its own layer (1,023) and the $K$ readout units; 2 state variables (voltage and adaptation) per neuron per window |

### 5.6 A worked example

OB 2x128 on SHD at 25 windows, worst case. A test recording has on average $N = 3{,}806.27$ non-empty cells; $T = 25$, $C = 700$, $W_1 = 128$, $P = 16{,}384$, $Q = 35{,}328$ ($K = 20$):

- reading the input, $T\,C\,m$: $25 \times 700 \times 1.25 = 21{,}875$ pJ
- first layer, $N\,W_1\,(3m + c)$: $3{,}806.27 \times 128 \times 3.98 = 1{,}939{,}081$ pJ
- other layers, $(T\,P + Q)\,(4m + c)$: $(25 \times 16{,}384 + 35{,}328) \times 5.23 = 444{,}928 \times 5.23 = 2{,}326{,}973$ pJ
- total: $4{,}287{,}929$ pJ, about $4.29\ \mu\text{J}$ per recording, the value the analysis reports for this design

## 6. What these numbers can and cannot say

- They are computed, not measured: no chip was built or measured.
- Every model is charged on the same imagined chip, so comparisons between models within one case are fair; the absolute values are indicative only.
- The accounting covers on-chip memory and arithmetic for 8-bit numbers. It leaves out off-chip memory, idle and leakage power, and control logic.
- The spiking neurons here are leaky (LIF and RadLIF). Dampfhoffer et al. found the simpler integrate-and-fire neuron more energy-efficient than leaky ones.
- The formulas in Section 5 are the ones implemented in the training notebooks (02_train_SHD and 03_train_SSC, Step 8), which follow the approach of Dampfhoffer et al.

## References

1. M. Dampfhoffer, T. Mesquida, A. Valentian and L. Anghel, Are SNNs Really More Energy-Efficient Than ANNs? An In-Depth Hardware-Aware Study, IEEE Transactions on Emerging Topics in Computational Intelligence 7(3), 731-741 (2022), doi:10.1109/TETCI.2022.3214509.
2. M. Horowitz, Computing's energy problem (and what we can do about it), IEEE International Solid-State Circuits Conference (ISSCC) 2014, Digest of Technical Papers, 10-14, doi:10.1109/ISSCC.2014.6757323.
3. A. Bittar and P. N. Garner, A surrogate gradient spiking baseline for speech command recognition, Frontiers in Neuroscience 16, 865897 (2022), doi:10.3389/fnins.2022.865897.
4. B. Cramer, Y. Stradmann, J. Schemmel and F. Zenke, The Heidelberg Spiking Data Sets for the Systematic Evaluation of Spiking Neural Networks, IEEE Transactions on Neural Networks and Learning Systems 33(7), 2744-2757 (2022), doi:10.1109/TNNLS.2020.3044364.
5. J. K. Eshraghian et al., Training Spiking Neural Networks Using Lessons From Deep Learning, Proceedings of the IEEE 111(9), 1016-1054 (2023), doi:10.1109/JPROC.2023.3308088.
