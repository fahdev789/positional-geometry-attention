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

This assumes:

* Content similarity drives interaction

* Position is injected externally

However, this separation creates:

* Redundant computation
* Global Dependency even when unnecessary

This work investigates **position-content entanglement** as a way to:

* Reduce attention scope
* Constraint attention locality
* Improve interpretability

# Key Idea

Instead of treating position as metadata, **we bind position into directly token representation:**

x = R(p) . e
Where:

𝑒
e = token embedding

𝑝
p = position index

𝑅
(
𝑝
)
R(p) = position-dependent transformation (e.g. rotation)

This changes attention from:

`"compare tokens globally"`

to:

`"compare position-aware features"`

# Experiments

**1. Attention without positional encoding**
* Identical token embeddings
* Permutation of tokens changes **nothing structurally.**
* Attention scores are permutation-invariant.

Observed behavior:
* Scores depend only on embedding similarity
* Order information lost

**2. Attention with Positional Encoding**
* Position alters embedding geometry
* Permutation changed dot-product

Observed behavior:
* Tokens embed position-aware similarity
* Attenstion becomes **order-aware by construction.**

# **3. Token permutation analysis**
We explicit permute token order **after score computation** to isolate:
* Content similarity
* Structural alignment

Permutation Reveals:
* Attention is not about tokens alone
* It is about **token-position aligment**

# Why Permutation Matters
Permutation is used as **a diagnostic tool.**
If attention fully understands the structure:
* Permuting tokens should disrupt alignment

If it does not:
* Score remain statistically similar

This repos shows:
* Without positional-binding *->* Permutation-invariant
* With positional-binding *->* Permutation-sensitive

This distinction is central.

# Scope and Non-Goals
** this is not:
* A production model
* a benchmark paper
* a performance optimization claim

**This is**
* mechanistic explorability
* research log
* A foundation of constrained-attention architecture

# Status
* A concept validated via a small-scale numerical examples
* Repo being built incrementaly file-by-file

# License
MIT License. This research documents **independent experiment and intrepretation** of publicaly known transformers mechanism.
