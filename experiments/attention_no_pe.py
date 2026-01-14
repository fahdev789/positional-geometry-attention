import numpy as np

np.set_printoptions(precision=4, suppress=True)

# ----------------------------
# Toy token embeddings (content only)
# ----------------------------
tokens = ["How", "does", "a", "model", "understands", "user", "goal"]

# Fixed embeddings (no position)
E = np.array([
    [0.2, 0.1, 0.4, 0.3],
    [0.1, 0.3, 0.2, 0.4],
    [0.4, 0.2, 0.1, 0.3],
    [0.3, 0.4, 0.2, 0.1],
    [0.1, 0.4, 0.3, 0.2],
    [0.2, 0.3, 0.4, 0.1],
    [0.3, 0.1, 0.2, 0.4],
])

# Identity projections (to isolate behavior)
Q = E
K = E
V = E

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
E_perm = E[perm]

out2, scores2, weights2 = attention(E_perm, E_perm, E_perm)

print("Original attention scores (row 0):")
print(scores1[0])

print("\nPermuted attention scores (row 0):")
print(scores2[0])
