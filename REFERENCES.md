# References

Every source below was opened to check the details given. For Horowitz (2014), the bibliographic record
and secondary copies of its cost figures were checked, not the original paper.

## Datasets

1. B. Cramer, Y. Stradmann, J. Schemmel and F. Zenke. The Heidelberg Spiking Data Sets for the
   Systematic Evaluation of Spiking Neural Networks. *IEEE Transactions on Neural Networks and Learning
   Systems* 33(7), 2744-2757 (2022). doi:10.1109/TNNLS.2020.3044364. Preprint: arXiv:1910.07407.
   (SHD and SSC; the LSTM set-up of `lstm_paper`.)
2. Zenke Lab. Spiking Heidelberg Datasets, download folders. https://zenkelab.org/datasets/
3. A. Amir et al. A Low Power, Fully Event-Based Gesture Recognition System. *Proceedings of the IEEE
   Conference on Computer Vision and Pattern Recognition (CVPR)*, 7243-7252 (2017). (DVS128 Gesture,
   used in the exploration notebook.)
4. tonic library, DVSGesture dataset (the version with one `.npy` file per gesture).
   https://tonic.readthedocs.io/en/latest/generated/tonic.datasets.DVSGesture.html

## Spiking networks

5. A. Bittar and P. N. Garner. A surrogate gradient spiking baseline for speech command recognition.
   *Frontiers in Neuroscience* 16, 865897 (2022). doi:10.3389/fnins.2022.865897. (RadLIF and its
   training recipe.)
6. A. Bittar. SpArch: Spiking Architectures for Speech Technology. Idiap Research Institute.
   https://github.com/idiap/sparch (BSD-3-Clause licence; the RadLIF code is adapted from it.)
7. J. K. Eshraghian et al. Training Spiking Neural Networks Using Lessons From Deep Learning.
   *Proceedings of the IEEE* 111(9), 1016-1054 (2023). doi:10.1109/JPROC.2023.3308088. (snnTorch, whose
   leaky and recurrent leaky neurons the small spiking networks follow.)
8. E. O. Neftci, H. Mostafa and F. Zenke. Surrogate Gradient Learning in Spiking Neural Networks.
   arXiv:1901.09948 (2019).

## LSTM and training methods

9. S. Hochreiter and J. Schmidhuber. Long Short-Term Memory. *Neural Computation* 9(8), 1735-1780
   (1997). doi:10.1162/neco.1997.9.8.1735.
10. F. A. Gers, J. Schmidhuber and F. Cummins. Learning to Forget: Continual Prediction with LSTM.
    *Neural Computation* 12(10), 2451-2471 (2000). doi:10.1162/089976600300015015.
11. R. Jozefowicz, W. Zaremba and I. Sutskever. An Empirical Exploration of Recurrent Network
    Architectures. *Proceedings of the 32nd International Conference on Machine Learning*, PMLR 37,
    2342-2350 (2015). (The forget-gate bias of 1.)
12. Y. Gal and Z. Ghahramani. A Theoretically Grounded Application of Dropout in Recurrent Neural
    Networks. arXiv:1512.05287 (2015). (Listed by Keras for its recurrent dropout.)
13. X. Glorot and Y. Bengio. Understanding the difficulty of training deep feedforward neural networks.
    *Proceedings of AISTATS 2010*, PMLR 9, 249-256 (2010). (Glorot-uniform initialisation.)
14. A. M. Saxe, J. L. McClelland and S. Ganguli. Exact solutions to the nonlinear dynamics of learning
    in deep linear neural networks. arXiv:1312.6120 (2013). (Orthogonal initialisation.)
15. Keras 2.3.0, LSTM layer source code (default initialisers and forget-gate bias).
    https://github.com/keras-team/keras/blob/2.3.0/keras/layers/recurrent.py
16. D. P. Kingma and J. Ba. Adam: A Method for Stochastic Optimization. *International Conference on
    Learning Representations (ICLR)* (2015). arXiv:1412.6980. (Adam and Adamax.)
17. S. Ioffe and C. Szegedy. Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift. arXiv:1502.03167 (2015).

## Order-blind designs

18. M. Zaheer, S. Kottur, S. Ravanbakhsh, B. Poczos, R. Salakhutdinov and A. Smola. Deep Sets.
    *Advances in Neural Information Processing Systems 30 (NIPS 2017)*. arXiv:1703.06114. (Order-blind
    functions as rho(sum of phi(x)); the basis of every order-blind design.)
19. S. Ravanbakhsh, J. Schneider and B. Poczos. Deep Learning with Sets and Point Clouds.
    arXiv:1611.04500 (2016).
20. C. R. Qi, H. Su, K. Mo and L. J. Guibas. PointNet: Deep Learning on Point Sets for 3D Classification
    and Segmentation. *CVPR 2017*, 652-660. (Pooling by maximum.)
21. L. Wang et al. Temporal Segment Networks: Towards Good Practices for Deep Action Recognition.
    arXiv:1608.00859 (2016). (Averaging the scores of parts of a recording, like `perminv`.)
22. E. Wagstaff, F. Fuchs, M. Engelcke, I. Posner and M. Osborne. On the Limitations of Representing
    Functions on Sets. arXiv:1901.09006 (2019). (Why the width of a design matters.)
23. R. L. Murphy, B. Srinivasan, V. Rao and B. Ribeiro. Janossy Pooling. arXiv:1811.01900 (2018).

## Energy

24. M. Dampfhoffer, T. Mesquida, A. Valentian and L. Anghel. Are SNNs Really More Energy-Efficient Than
    ANNs? An In-Depth Hardware-Aware Study. *IEEE Transactions on Emerging Topics in Computational
    Intelligence* 7(3), 731-741 (2023; online 2022). doi:10.1109/TETCI.2022.3214509.
25. M. Horowitz. Computing's energy problem (and what we can do about it). *IEEE International
    Solid-State Circuits Conference (ISSCC) 2014, Digest of Technical Papers*, 10-14.
    doi:10.1109/ISSCC.2014.6757323.

## Related work on whether benchmarks need timing

26. C. Ma et al. Spiking Neural Networks for Temporal Processing: Status Quo and Future Prospects.
    arXiv:2502.09449 (2025).
