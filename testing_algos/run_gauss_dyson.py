# training
import pandas as pd
import numpy as np
import numba as nb
from tqdm import tqdm
import sys

NUM_POTENTIALS = 10**5

def precompute_unit_vectors(num_vecs, dim, seed=None):
    if seed is not None:
        np.random.seed(seed)
    vecs = 2 * np.random.rand(num_vecs, dim) - 1
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms

def get_data(npz_file): return np.load(npz_file, allow_pickle=True).item()

@nb.njit(fastmath=True)
def round_scaled(vec, scale):
    return np.round(vec / scale) * scale

@nb.njit(fastmath=True)
def random_vector(n):
    return np.random.uniform(0, 100, size=n)

@nb.njit(fastmath=True)
def gen(vec, scale):
    rdm = round_scaled(random_vector(len(vec)), scale)
    helper = rdm - vec
    return helper, rdm

@nb.njit(fastmath=True)
def recov(helper, vec, scale):
    return round_scaled(helper + vec, scale)

@nb.njit(parallel=True, fastmath=True)
def match(c_vec, q_vec, scale, unit_vectors):
    OFFSET = scale

    helper, a = gen(c_vec, 0.70892333984375 * 0.72)
    b = recov(helper, q_vec, 0.70892333984375 * 0.72)

    if np.array_equal(a, b):
        return True

    found = 0
    small_lat_sc = 0.4

    for i in nb.prange(unit_vectors.shape[0]):
        vec = unit_vectors[i]
        ofc = c_vec + vec

        helper, a = gen(ofc * OFFSET * 0.95, OFFSET * small_lat_sc)
        b = recov(helper, q_vec, OFFSET * small_lat_sc)

        if np.array_equal(a, b):
            found = 1   # SAFE: monotonic write

    return found == 1

    

def run_bin_search(data, coeff, unit_vectors):

    keys = list(data.keys())

    tchk, fchk = 0, 0
    tks, fks = 0, 0

    for key in tqdm(keys):
        rad = data[key][1] * coeff
        cen = data[key][0]

        tchk += sum([1 if match(cen, val, rad, unit_vectors) else 0 for val in data[key][2]])
        tks += len(data[key][2])
        fchk += sum([1 if match(cen, val, rad, unit_vectors) else 0 for val in data[key][3]])
        fks += len(data[key][3])

    tmr, fmr = tchk / tks, fchk / fks
    
    return tmr, fmr

def run_sweep(data, save_path, COEFF, unit_vectors):
    
    res_ma = {"coeff": [], "TMR": [], "FMR": []}

    tmr, fmr = run_bin_search(data, COEFF, unit_vectors)

    res_ma["coeff"].append(COEFF)
    res_ma["TMR"].append(tmr)
    res_ma["FMR"].append(fmr)
    
    print("Done! Check:", save_path)
    pd.DataFrame.from_dict(res_ma, orient='columns').to_csv(save_path, index=False)

def main():
    src_file = sys.argv[1]
    dst_file = sys.argv[2]
    coeff = float(sys.argv[3])

    data = get_data(src_file)

    unit_vectors = precompute_unit_vectors(
        num_vecs=10**5,
        dim=64,
        seed=42
    )

    run_sweep(data, dst_file, coeff, unit_vectors)

main()
