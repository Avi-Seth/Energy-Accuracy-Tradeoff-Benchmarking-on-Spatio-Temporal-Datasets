# References

Numbered as in `References_How_They_Are_Used.docx`. Each entry ends with what the project takes from it.
Bibliographic details were checked against publisher or index records. For Horowitz (2014), the
original paper was not checked; its 8-bit cost figures were checked in a secondary copy.

## Data

1. B. Cramer, Y. Stradmann, J. Schemmel and F. Zenke. The Heidelberg Spiking Data Sets for the
   Systematic Evaluation of Spiking Neural Networks. *IEEE Transactions on Neural Networks and Learning
   Systems* 33(7), 2744-2757 (2022; published online 2020). doi:10.1109/TNNLS.2020.3044364.
   Data: https://zenkelab.org/datasets/ (Creative Commons Attribution 4.0 International).
   (SHD and SSC; the LSTM set-up of `lstm_paper`.)
2. A. Amir et al. A Low Power, Fully Event-Based Gesture Recognition System. *2017 IEEE Conference on
   Computer Vision and Pattern Recognition (CVPR)*, 7388-7397 (2017). doi:10.1109/CVPR.2017.781.
   Downloaded with the tonic library:
   https://tonic.readthedocs.io/en/latest/generated/tonic.datasets.DVSGesture.html
   (DVS128 Gesture; exploration notebook only, off by default.)

## Spiking networks

3. A. Bittar and P. N. Garner. A surrogate gradient spiking baseline for speech command recognition.
   *Frontiers in Neuroscience* 16, 865897 (2022). doi:10.3389/fnins.2022.865897.
   (RadLIF and its training recipe.)
4. SpArch: Spiking Architectures for Speech Technology. Idiap Research Institute.
   https://github.com/idiap/sparch. Copyright 2022 Idiap Research Institute, BSD-3-Clause licence.
   (The RadLIF code in `02_train_SHD.ipynb` is adapted from it; its notice is kept in the notebook.)
5. J. K. Eshraghian et al. Training Spiking Neural Networks Using Lessons From Deep Learning.
   *Proceedings of the IEEE* 111(9), 1016-1054 (2023). doi:10.1109/JPROC.2023.3308088.
   (snnTorch: the neurons of SNN-FF 128 and SNN-R 128 follow snnTorch 1.0.0, checked in Step 11.)
6. E. O. Neftci, H. Mostafa and F. Zenke. Surrogate Gradient Learning in Spiking Neural Networks:
   Bringing the Power of Gradient-Based Optimization to Spiking Neural Networks. *IEEE Signal
   Processing Magazine* 36(6), 51-63 (2019). doi:10.1109/MSP.2019.2931595.
   (How the spiking networks are trained.)

## LSTM and training

7. S. Hochreiter and J. Schmidhuber. Long Short-Term Memory. *Neural Computation* 9(8), 1735-1780
   (1997). doi:10.1162/neco.1997.9.8.1735. (The LSTM.)
8. Keras 2.3.0, LSTM layer source code.
   https://github.com/keras-team/keras/blob/2.3.0/keras/layers/recurrent.py
   (The defaults written out in `lstm_paper`: initialisers, forget-gate bias of 1, recurrent dropout.
   Its documentation lists the papers behind them.)
9. D. P. Kingma and J. Ba. Adam: A Method for Stochastic Optimization. *International Conference on
   Learning Representations (ICLR)* (2015). arXiv:1412.6980. (Adam for SHD, Adamax for SSC.)
10. S. Ioffe and C. Szegedy. Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift. *Proceedings of the 32nd International Conference on Machine Learning*,
    PMLR 37, 448-456 (2015). (Used in RadLIF 2x1,024 and OB 2x1,024.)

## Order-blind designs

11. M. Zaheer, S. Kottur, S. Ravanbakhsh, B. Poczos, R. Salakhutdinov and A. Smola. Deep Sets.
    *Advances in Neural Information Processing Systems 30* (2017). arXiv:1703.06114.
    (The form of every order-blind design, and so of the order-blind ceiling.)
12. C. R. Qi, H. Su, K. Mo and L. J. Guibas. PointNet: Deep Learning on Point Sets for 3D
    Classification and Segmentation. *2017 IEEE Conference on Computer Vision and Pattern Recognition
    (CVPR)*, 77-85 (2017). doi:10.1109/CVPR.2017.16. (Combining the windows by their maximum.)

## Energy

13. M. Dampfhoffer, T. Mesquida, A. Valentian and L. Anghel. Are SNNs Really More Energy-Efficient Than
    ANNs? An In-Depth Hardware-Aware Study. *IEEE Transactions on Emerging Topics in Computational
    Intelligence* 7(3), 731-741 (2023; published online 2022). doi:10.1109/TETCI.2022.3214509.
    (The energy question and the accounting method in Step 8.)
14. M. Horowitz. Computing's energy problem (and what we can do about it). *2014 IEEE International
    Solid-State Circuits Conference (ISSCC), Digest of Technical Papers*, 10-14 (2014).
    doi:10.1109/ISSCC.2014.6757323. (The 45 nm costs of 8-bit memory reads and arithmetic.)
