# Attention Under Positional Constraints

# Overview

This repository explores attention mechanisms under constrained settings, focusing on how positional information can be entangled with token content to reduce reliance on full global self-attention. The work is experimental, mechanistic, and grounded in first-principles analysis of transformers.

The core question:

*`If position and content are already jointly encoded, how much explicit attention is actually required?`*

This repo documents controlled experiments comparing:

* Attention **without positional encoding**

* Attention **with positional encoding (additive, multiplicative, rotational)**

* Effects of **token permutation** on attention scores

The goal is not **performance benchmarking,** but **mechanistic understanding.**

# Motivation

Standard transformers rely on:

Attention (Q, K, V) = softmax(Q.K^T / sqrt(d))*V
