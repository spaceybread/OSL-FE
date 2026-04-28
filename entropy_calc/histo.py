from collections import defaultdict
from multiprocessing import Pool, cpu_count
import numpy as np
import sys
from tqdm import tqdm

INNER_LATTICE_SCALE = 0.2
OUTER_LATTICE_SCALE = 0.45
RING_RADIUS = 0.9
NUM_POTENTIALS = 10**4

def precompute_unit_vectors(num_vecs, dim, seed=42):
    np.random.seed(seed)
    vecs = 2 * np.random.rand(num_vecs, dim) - 1
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return (vecs / norms).astype(np.float32)


def decode_vectorized(w, unit_vectors):
    inner = (np.round(w / INNER_LATTICE_SCALE) * INNER_LATTICE_SCALE)[None, :]

    deltas = unit_vectors * RING_RADIUS
    w_shifted = w[None, :] - deltas
    outer = np.round(w_shifted / OUTER_LATTICE_SCALE) * OUTER_LATTICE_SCALE

    return np.vstack([inner, outer])


def process_user(args):
    vecs, unit_vectors = args
    local_counts = defaultdict(int)
    for w in vecs:
        candidates = decode_vectorized(w.astype(np.float32), unit_vectors)
        rounded = np.round(candidates, 6)
        for row in rounded:
            local_counts[row.tobytes()] += 1
    return local_counts


def merge_counts(results):
    merged = defaultdict(int)
    for local in results:
        for k, v in local.items():
            merged[k] += v
    return merged


def build_histogram(user_vecs, unit_vectors, n_workers):
    args = [(np.array(vecs, dtype=np.float32), unit_vectors)
            for vecs in user_vecs.values()]

    with Pool(processes=n_workers) as pool:
        results = list(tqdm(pool.imap(process_user, args),
                            total=len(args),
                            desc="Building histogram"))

    return merge_counts(results)


def compute_entropy(hist):
    counts = np.array(list(hist.values()), dtype=np.float64)
    total = counts.sum()
    probs = counts / total
    return -np.sum(probs * np.log2(probs + 1e-12))


def main():
    if len(sys.argv) < 2:
        print("Usage: python estimate_entropy.py <user_vecs.npy> [num_workers]")
        sys.exit(1)

    npy_file = sys.argv[1]
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else cpu_count()

    print(f"Loading {npy_file}...")
    user_vecs = np.load(npy_file, allow_pickle=True).item()

    sample = list(user_vecs.values())[0]
    dim = sample.shape[1]
    print(f"Users: {len(user_vecs)} | Dim: {dim} | Workers: {n_workers}")

    print(f"Precomputing {NUM_POTENTIALS} unit vectors in {dim}d...")
    unit_vectors = precompute_unit_vectors(NUM_POTENTIALS, dim, seed=42)

    hist = build_histogram(user_vecs, unit_vectors, n_workers)

    entropy = compute_entropy(hist)
    total_vecs = sum(len(v) for v in user_vecs.values())

    print(f"\n--- Results ---")
    print(f"Total vectors:    {total_vecs}")
    print(f"Unique bins:      {len(hist)}")
    print(f"Fuzzy entropy:    {entropy:.4f} bits")

    sorted_hist = sorted(hist.items(), key=lambda x: x[1], reverse=True)
    print(f"\nTop 5 most common lattice centers:")
    for k, v in sorted_hist[:5]:
        center = np.frombuffer(k, dtype=np.float32)
        print(f"  count={v}  center={center[:4]}...  (first 4 dims)")


if __name__ == "__main__":
    main()
