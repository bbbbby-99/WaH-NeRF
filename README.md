# WaH-NeRF：Where and How: Mitigating Confusion in Neural Radiance Fields from Sparse Inputs
The official implementation of ACM MM 2023 paper. [[Paper]](https://arxiv.org/pdf/2308.02908.pdf)

Yanqi Bao, Yuxin Li, Jing Huo, Tianyu Ding, Xinyue Liang, Wenbin Li and Yang Gao

## Introduction
Neural Radiance Fields from Sparse inputs (NeRF-S) have shown great potential in synthesizing novel views with a limited number of observed viewpoints. However, due to the inherent limitations of sparse inputs and the gap between non-adjacent views, rendering results often suffer from over-fitting and foggy surfaces, a phenomenon we refer to as "CONFUSION" during volume rendering. In this paper, we analyze the root cause of this confusion and attribute it to two fundamental questions: "WHERE" and "HOW". To this end, we present a novel learning framework, WaH-NeRF, which effectively mitigates confusion by tackling the following challenges: (i) “WHERE” to Sample? in NeRF-S---we introduce a Deformable Sampling strategy and a Weight-based Mutual Information Loss to address sample-position confusion arising from the limited number of viewpoints; and (ii) “HOW” to Predict? in NeRF-S---we propose a Semi-Supervised NeRF learning Paradigm based on pose perturbation and a Pixel-Patch Correspondence Loss to alleviate prediction confusion caused by the disparity between training and testing viewpoints. By integrating our proposed modules and loss functions, WaH-NeRF outperforms previous methods under the NeRF-S setting.
![Pipeline in WaHNeRF](https://github.com/bbbbby-99/WaH-NeRF/blob/main/image/framework.png)

## Installation
  ```js
  git clone https://github.com/bbbbby-99/WaH-NeRF.git
  cd WaH-NeRF
  ```
