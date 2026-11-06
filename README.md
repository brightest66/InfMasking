# InfMasking: A Unified Variational Framework for Synergistic Multimodal Learning

This repository contains the official implementations of **InfMasking**, including our **NeurIPS 2025 Spotlight** paper and its extended journal version, **accepted by IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2026**.

| Version                            | Paper                                                                                     | Links                                                                                                                 | Implementation                                                                                     |
| ---------------------------------- | ----------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| **TPAMI 2026**               | **InfMasking: A Unified Variational Framework for Synergistic Multimodal Learning** | [DOI](https://doi.org/10.1109/TPAMI.2026.3751348) | Unsupervised code in the repository root; supervised extension in [TPAMI2026/](TPAMI2026/README.md) |
| **NeurIPS 2025 (Spotlight)** | InfMasking: Unleashing Synergistic Information by Contrastive Multimodal Interactions     | [arXiv](https://arxiv.org/abs/2509.25270)                                                                              | Unsupervised experiments in the repository root                                                    |

For the supervised experiments introduced in the journal extension, see the [TPAMI 2026 instructions](TPAMI2026/README.md). The [training and evaluation instructions below](#training-and-evaluation-neurips-2025) cover the original unsupervised experiments.

## Overview

InfMasking is a unified multimodal learning framework based on **Infinite Masking**. It randomly masks modality-specific feature tokens before fusion, exposing the fusion module to diverse partial combinations of cross-modal evidence. Requiring the masked fused representations to remain consistent with a reference target encourages the model to capture synergistic interactions and reduces reliance on a dominant modality or shared information alone.

- **Unsupervised learning:** align masked fused views with their unmasked counterparts through a contrastive objective.
- **Supervised learning:** use class proxies as the reference targets, combining the label-proxy InfMasking objective with fused and unimodal classification losses.

The expectation over infinitely many masks is approximated using closed-form variational bounds under a Gaussian model of masked fused representations: a lower bound for the unsupervised objective and an upper bound for the supervised objective.

<div align="center">
  <img src="TPAMI2026/assets/flow_chart.png" alt="Unified InfMasking framework for supervised and unsupervised multimodal learning." width="100%" />
</div>

## Extensions over the NeurIPS 2025 Paper

The TPAMI journal paper substantially extends the conference version in four aspects:

1. **Unified formulation:** a single Infinite Masking framework for unsupervised and supervised learning, including a label-proxy masked fusion objective.
2. **Supervised variational objective:** a tractable upper bound under a Gaussian model of masked fusion states, extending the conference analysis beyond the unsupervised setting.
3. **Tightness analysis:** covariance-controlled gap guarantees for both variational bounds under the Gaussian model, together with Monte Carlo evaluation of the surrogate gaps and analysis of conditions beyond Gaussianity.
4. **Broader experimental validation:** supervised experiments on MOSI, MOSEI, MIMIC, MAMI and MUTE, with additional ablations, robustness evaluation, PID estimates and InterSHAP interaction analysis.

## Citation

If you find this repository useful for your research, please cite the relevant paper(s).

**Journal version (TPAMI 2026):**

```bibtex
@article{wen2026infmasking,
  title={InfMasking: A Unified Variational Framework for Synergistic Multimodal Learning},
  author={Liangjian Wen and Qun Dai and Jiang Duan and Yong Dai and Jianzhuang Liu and Dongkai Wang and Jun Wang and Fuyuan Cao and Zenglin Xu and Zhao Kang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2026},
  doi={10.1109/TPAMI.2026.3751348},
  url={https://doi.org/10.1109/TPAMI.2026.3751348}
}
```

**Conference version (NeurIPS 2025):**

```bibtex
@inproceedings{NEURIPS2025_16fe2047,
 author = {Wen, Liangjian and Dai, Qun and Liu, Jianzhuang and Zheng, Jiangtao and Dai, Yong and Wang, Dongkai and Kang, Zhao and Wang, Jun and Xu, Zenglin and Duan, Jiang},
 booktitle = {Advances in Neural Information Processing Systems},
 doi = {10.52202/085713-0525},
 editor = {D. Belgrave and C. Zhang and H. Lin and R. Pascanu and P. Koniusz and M. Ghassemi and N. Chen},
 pages = {15529--15555},
 publisher = {Curran Associates, Inc.},
 title = {InfMasking: Unleashing Synergistic Information  by Contrastive Multimodal Interactions},
 url = {https://proceedings.neurips.cc/paper_files/paper/2025/file/16fe204707eac76cfd0e94296fe98cef-Paper-Conference.pdf},
 volume = {38, Main Conference},
 year = {2025}
}
```

## Training and Evaluation (NeurIPS 2025)

The commands and result tables in this section describe the **unsupervised** experiments. For the supervised TPAMI 2026 extension, use the separate environment and commands in [TPAMI2026/README.md](TPAMI2026/README.md).

### Installation

Please create a conda environment with python 3.8 and install the required packages after the code is cloned.

```bash
conda env create -f environment.yml
conda activate multimodal
cd InfMasking
```

### Synthetic Experiments on Trifeature Datasets

To run the synthetic experiments on Trifeature datasets, please execute the following command:

```bash
bash run_scripts/trifeatures_run.sh
```

To evaluate the learned representations, please execute the following command. Please change the dataset and specific-dataset parameters in `run_scripts/test.sh` before running the command.

```bash
bash run_scripts/test.sh
```
The evaluation of below models is supported in `run_scripts/test.sh`.

#### Results

We select "Cross", "Cross+Self", FactorCL, MAE and CoMM as baselines for comparison.
Linear probing accuracy (in \%) of redundancy (shape), uniqueness (texture) and synergy (color and texture) on Trifeature dataset is shown in the table below.

| Model                           | redundancy↑        | uniqueness↑                 | synergy↑                    |
| ------------------------------- | ------------------- | ---------------------------- | ---------------------------- |
| $\text{Cross}^\clubsuit$      | **$100.0$** | $11.6$                     | $50.0$                     |
| $\text{Cross+Self}^\clubsuit$ | $99.7$            | $86.9$                     | $50.0$                     |
| $\text{FactorCL}^\clubsuit$   | $99.8$            | $62.5$                     | $46.5$                     |
| $\text{MAE}$                  | $99.8_{\pm0.11}$  | $82.4_{\pm3.09}$           | $50.1_{\pm0.24}$           |
| $\text{CoMM}$                 | $99.9_{\pm0.06}$  | $86.8_{\pm2.99}$           | $71.4_{\pm3.47}$           |
| $\text{InfMasking (ours)}$    | $99.9_{\pm0.09}$  | **$90.6_{\pm2.31}$** | **$77.0_{\pm4.22}$** |

$^\clubsuit$ Results are taken from the paper "[What to align in multimodal contrastive learning ?](https://arxiv.org/abs/2409.07402)".

### Experiments with 2 Modalities on Multibench

To run the experiments with 2 modalities on Multibench, please execute the following command:

```bash
bash run_scripts/multibench_run.sh
```

#### Results

Linear probing MSE($\times 10^{-4}$) for regression task and top-1 accuracy (in \%) for classification tasks on Multibench is shown in the table below.

| Model                           | RegressionV&TEE↓            | MIMIC↑                      | MOSI↑                      | UR-FUNNY↑                  | MUSTARD↑                   | ClassificationAverage\* ↑ |
| ------------------------------- | ---------------------------- | ---------------------------- | --------------------------- | --------------------------- | --------------------------- | -------------------------- |
| $\text{Cross}^\clubsuit$      | $33.09_{\pm3.67}$          | $66.7_{\pm0.1}$            | $47.8_{\pm1.8}$           | $50.1_{\pm1.9}$           | $53.5_{\pm2.9}$           | $54.52$                  |
| $\text{Cross+Self}^\clubsuit$ | $7.56_{\pm0.31}$           | $65.49_{\pm0.0}$           | $49.0_{\pm1.1}$           | $59.9_{\pm0.9}$           | $53.9_{\pm4.0}$           | $57.07$                  |
| $\text{FactorCL}^\clubsuit$   | $10.82_{\pm0.56}$          | $67.3_{\pm0.0}$            | $51.2_{\pm1.6}$           | $60.5_{\pm0.8}$           | $55.80_{\pm0.9}$          | $58.7$                   |
| $\text{CoMM}$                 | $7.96_{\pm2.13}$           | $66.4_{\pm0.41}$           | $63.7_{\pm2.5}$           | $63.3_{\pm0.51}$          | $64.4_{\pm1.1}$           | $64.45$                  |
| $\text{InfMasking (ours)}$    | **$4.23_{\pm0.37}$** | **$68.1_{\pm0.42}$** | **$69.0_{\pm1.2}$** | **$64.3_{\pm0.9}$** | **$66.8_{\pm2.5}$** | **$67.05$**        |

$^\clubsuit$ Results are taken from the paper "[What to align in multimodal contrastive learning ?](https://arxiv.org/abs/2409.07402)".

### Experiments with 3 Modalities on Multibench

To run the experiments with 3 modalities on Multibench, please execute the following command:

```bash
bash run_scripts/all-mod_run.sh
```

#### Results

Linear probing top-1 accuracy (in \%) for classification tasks on Vision\&Touch and UR-FUNNY datasets with 3 modalities on Multibench is shown in the table below.

| Model                        | #Mod. | V&T CP↑                     | UR-FUNNY↑                   |
| ---------------------------- | ----- | ---------------------------- | ---------------------------- |
| $\text{Cross}$             | 2     | $86.3_{\pm0.25}$           | $50.1^\clubsuit$           |
| $\text{Cross+Self}$        | 2     | $87.6_{\pm0.26}$           | $59.9^\clubsuit$           |
| $\text{CoMM}$              | 2     | $87.0_{\pm1.77}$           | $63.3_{\pm0.51}$           |
| $\text{InfMasking (ours)}$ | 2     | $88.5_{\pm0.33}$           | $64.3_{\pm0.9}$            |
| $\text{CMC}^\clubsuit$     | 3     | $94.1$                     | $59.2$                     |
| $\text{CoMM}$              | 3     | **$94.1_{\pm0.17}$** | $64.8_{\pm1.13}$           |
| $\text{InfMasking (ours)}$ | 3     | $94.1_{\pm0.09}$           | **$65.6_{\pm1.15}$** |

$^\clubsuit$ Results are taken from the paper "[What to align in multimodal contrastive learning ?](https://arxiv.org/abs/2409.07402)".

### Experiments with 2 Modalities on Multimodal IMDb

To run the experiments with 2 modalities on Multimodal IMDb, please execute the following command:

```bash
bash run_scripts/MM_IMDb_run.sh
```

#### Results

Linear probing F1-score (weighted and macro) (in \%) for MM-IMDB is shown in the table below.

| Model                                                | Modalities | weighted-f1↑                 | macro-f1↑                    |
| ---------------------------------------------------- | ---------- | ----------------------------- | ----------------------------- |
| $\text{SimCLR}^{\clubsuit\triangle}$               | V          | $40.35_{\pm0.23}$           | $27.99_{\pm0.33}$           |
|                                                      | V          | $51.5$                      | $40.8$                      |
| $\text{CLIP}^\clubsuit$                            | L          | $51.0$                      | $43.0 $                       |
|                                                      | V+L        | $58.9_{\pm0.0}$             | $50.9_{\pm0.0}$             |
| $\text{SLIP}^{\clubsuit\triangle}$                 | V+L        | $56.54_{\pm0.19}$           | $47.35_{\pm0.27}$           |
| $\text{CLIP}^{\clubsuit\triangle}$                 | V+L        | $54.49_{\pm0.19}$           | $44.94_{\pm0.30}$           |
| $\text{CoMM}_{\text{(CLIP backbone)}}$             | V+L        | $61.29_{\pm0.73}$           | $53.79_{\pm0.22}$           |
| $\text{InfMasking}_{\text{(ours, CLIP backbone)}}$ | V+L        | **$62.60_{\pm0.26}$** | **$55.93_{\pm0.19}$** |

$^\triangle$ indicates further training on unlabeded data. $^\clubsuit$ Results are taken from the paper "[What to align in multimodal contrastive learning ?](https://arxiv.org/abs/2409.07402)".

## Thanks

We sincerely thank the contributors of the following repositories, which have been instrumental in our implementation:

- [What to align in multimodal contrastive learning ?](https://github.com/Duplums/CoMM)
- [Masked Autoencoders Are Scalable Vision Learners](https://github.com/facebookresearch/mae)
