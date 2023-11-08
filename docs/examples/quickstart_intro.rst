..
    Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Overview
--------

Transformer Engine (TE) is a library for accelerating Transformer models on NVIDIA GPUs, providing 
better performance with lower memory utilization in both training and inference. It provides 
support for 8-bit floating point (FP8) precision on Hopper GPUs, implements a collection of highly 
optimized building blocks for popular Transformer architectures, and exposes an 
automatic-mixed-precision-like API that can be used seamlessy with your PyTorch and Jax codes. It 
also includes a framework-agnostic C++ API that can be integrated with other deep learning 
libraries to enable FP8 support for Transformers.

Let's build a Transformer layer!
--------------------------------

Let's start with creating a GPT encoder layer. Figure 1 shows the overall structure.

.. raw:: html
    <figure align="center">
        <img src="transformer_layer.png" width="20%">
        <figcaption> Figure 1: Structure of a GPT encoder layer.</figcaption>
    </figure>