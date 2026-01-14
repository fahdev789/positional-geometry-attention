# Purpose: **show how positional encoding makes attention order-sensitive.**

import numpy as np

np.set_printoptions(precision=4, suppress=True)

# ----------------------------
# Tokens and base embeddings
# ----------------------------
tokens = ["How", "does", "a", "model", "understands", "user", "goal"]
E = np.array([
    [0.2, 0.1, 0.4, 0.3],
    [0.1, 0.3, 0.2, 0.4],
    [0.4, 0.2, 0.1, 0.3],
    [0.3, 0.4, 0.2, 0.1],
    [0.1, 0.4, 0.3, 0.2],
    [0.2, 0.3, 0.4, 0.1],
    [0.3, 0.1, 0.2, 0.4],
])

d = E.shape[1]
n = len(tokens)

# ----------------------------
# Sinusoidal positional encoding
# ----------------------------
def sinusoidal_pe(n, d):
    PE = np.zeros((n, d))
    for pos in range(n):
        for i in range(0, d, 2):
            PE[pos, i] = np.sin(pos / (10000 ** (i / d)))
            if i+1 < d:
                PE[pos, i+1] = np.cos(pos / (10000 ** (i / d)))
    return PE

PE = sinusoidal_pe(n, d)

# Add PE to embeddings
E_pe = E + PE

# Identity projections (for simplicity)
Q = E_pe
K = E_pe
V = E_pe

def attention(Q, K, V):
    scores = Q @ K.T
    weights = softmax(scores)
    return weights @ V, scores, weights

def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    return np.exp(x) / np.exp(x).sum(axis=-1, keepdims=True)

# ----------------------------
# Original order
# ----------------------------
out1, scores1, weights1 = attention(Q, K, V)

# ----------------------------
# Permuted order
# ----------------------------
perm = [0, 5, 6, 2, 1, 4, 3]
E_perm = E_pe[perm]

out2, scores2, weights2 = attention(E_perm, E_perm, E_perm)

print("Attention scores with PE (original, row 0):")
print(scores1[0])

print("\nAttention scores with PE (permuted, row 0):")
print(scores2[0])
