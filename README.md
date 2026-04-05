<h1 align="center">Seeing What Matters: Visual Preference Policy Optimization for Visual Generation </h1>
<div align="center">
  <!-- <a href='LICENSE'><img src='https://img.shields.io/badge/license-MIT-yellow'></a> -->
  <a href='https://arxiv.org/abs/2511.18719'><img src='https://img.shields.io/badge/ArXiv-red?logo=arxiv'></a>  &nbsp;
  <a href='https://github.com/ziqi-nox/ViPO'><img src="https://img.shields.io/badge/Code-9E95B7?logo=github"></a> &nbsp;
</div>


This repository provides the implementation of **ViPO (Visual Preference Policy Optimization)** for visual generation.

**Note.** This codebase is **built upon DanceGRPO** and follows its overall GRPO training pipeline and project organization. For model-specific components, this repository also relies on the official implementations and dependencies of **Wan2.1** and **VideoAlign**.

## Overview

Recent GRPO-based visual alignment pipelines usually optimize a **single scalar reward** for each image or video sample. ViPO lifts such coarse supervision into **structured visual preference signals**, enabling optimization to focus more on perceptually important spatial and temporal regions.

ViPO is designed to be:

- **Architecture-agnostic**: compatible with existing GRPO-style training pipelines.
- **Lightweight**: introduces structured preference maps without changing the overall training paradigm.
- **Applicable to both image and video generation**.

## Codebase Attribution

This repository is developed **based on [DanceGRPO](https://github.com/XueZeyue/DanceGRPO)**. We reuse and extend the training framework, and adapt it for **region-aware preference optimization**.

In particular:

- The **training framework and overall code organization** are based on DanceGRPO.
- The **Wan2.1-related generation components** follow the official [Wan2.1](https://github.com/Wan-Video/Wan2.1) implementation.
- The **video reward / evaluation related components** are adapted with reference to [VideoAlign](https://github.com/KlingAIResearch/VideoAlign).

## Installation

We recommend preparing the environment by following the official instructions of the upstream repositories:

1. **DanceGRPO** for the main GRPO training framework.
2. **Wan2.1** for Wan-based generation dependencies.
3. **VideoAlign** for reward-model / evaluation related dependencies.

Because ViPO is implemented on top of DanceGRPO, we recommend using **DanceGRPO as the base environment**, and then installing any additional dependencies required by Wan2.1 or VideoAlign for the features you actually use.

## Training
```bash
bash scripts/preprocess/preprocess_flux_rl_embeddings.sh # Preprocess
bash scripts/finetune/finetune_flux_vipo.sh # FLUX training
```
```bash
bash scripts/preprocess/preprocess_wan_rl_embedding.sh # Preprocess
bash scripts/finetune/finetune_wan_vipo.sh # Wan2.1 training
```

### Inference
```bash
torchrun --nproc_per_node=$GPU_NUM --master_port 19001 \ # Flux Inference
  scripts/visualization/vis_flux.py \
  --ft_dir /path/to/your/flux_checkpoint \
  --output_dir /path/to/output_dir
```
For Wan2.1 inference, please adapt the corresponding Wan2.1 / DiffSynth inference entry and set the checkpoint path accordingly.


## Acknowledgements
This repository is built upon or references the following excellent open-source projects:
- [DanceGRPO](https://github.com/XueZeyue/DanceGRPO)
- [Wan2.1](https://github.com/Wan-Video/Wan2.1)
- [VideoAlign](https://github.com/KlingAIResearch/VideoAlign)

We thank the authors and contributors of these projects for making their code and models publicly available.

## Citation

If you find ViPO useful for your research, please cite:

```bibtex
@article{ni2025seeing,
  title={Seeing What Matters: Visual Preference Policy Optimization for Visual Generation},
  author={Ni, Ziqi and Liang, Yuanzhi and Li, Rui and Zhou, Yi and Huang, Haibin and Zhang, Chi and Li, Xuelong},
  journal={arXiv preprint arXiv:2511.18719},
  year={2025}
}
```
