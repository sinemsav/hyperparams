# Federated Hyperparameter Tuning

[![arxiv](https://img.shields.io/badge/arXiv-2402.16087-b31b1b.svg)](https://arxiv.org/abs/2402.16087)

Implementation: Sinem Sav, Natalija Mitic, Simon Perriard Nicolas, and Xavier Oliva Jürgens

This repository contains code accompanying the paper titled _**PrivTuna: Efficient Hyperparameter Tuning for Privacy-Preserving Federated Learning**_. The paper addresses the challenge of privacy-preserving hyperparameter (HP) tuning in cross-silo federated learning (FL). We conduct experiments using various datasets and model architectures, involving different numbers of clients and data distributions, with the aim of better understanding the relationship between the HPs discovered by each client and the server ones.

## Repository Structure

The repository is organized into two main folders:

1. [`IID_setting`](./IID_setting/): Contains code and instructions for experiments conducted under the IID (independent and identically distributed) data setting.
2. [`non-IID_setting`](./non_IID_setting/): Contains code and instructions for experiments conducted under the non-IID (non-independent and identically distributed) data setting.

## Usage

Detailed explanations, requirements and instructions for running experiments can be found within each respective folder: [`IID_setting`](./IID_setting/), [`non-IID_setting`](./non_IID_setting/).

Please ensure that all dependencies are installed before running the code.

## Paper Abstract

In the paper, we address the problem of privacy-preserving hyperparameter (HP) tuning in cross-silo federated learning (FL). We first perform a comprehensive measurement study on various datasets and model architectures, with varying numbers of clients and data distributions, aiming to better understand the relationship between the HPs discovered by each client and the server ones. We observe that the optimal parameters of the FL server, e.g., the learning rate, can be accurately and efficiently tuned based on the HPs found by each client on its local data: (i) HP averaging is suitable for iid settings, while (ii) density-based clustering can uncover the optimal set of parameters in non-iid ones. Then, to prevent information leakage from the exchange of the clients' local HPs, we design and implement **PrivTuna**, a novel framework for privacy-preserving hyperparameter tuning using multi-party homomorphic encryption. We use **PrivTuna** to implement privacy-preserving federated averaging and density-based clustering, and we experimentally evaluate its performance demonstrating its computation/communication efficiency and its precision in tuning hyperparameters.
