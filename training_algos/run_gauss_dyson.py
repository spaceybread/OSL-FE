# training
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
import numba as nb
from tqdm import tqdm
import sys

# from dyson_vggface_spec import * 
# from dyson_ytf_spec import *  
# from dyson_voice_spec import * 

UNIT_VECTORS = None

NUM_POTENTIALS = 10**4
RING_RADIUS = 0
INNER_LATTICE_SCALE = 0.2
OUTER_LATTICE_SCALE = 0


def init_worker(unit_vectors):
    global UNIT_VECTORS, RING_OFFSETS
    UNIT_VECTORS = unit_vectors
    RING_OFFSETS = unit_vectors * RING_RADIUS

def precompute_unit_vectors(num_vecs, dim, seed=None):
    if seed is not None:
        np.random.seed(seed)
    vecs = 2 * np.random.rand(num_vecs, dim) - 1
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return (vecs / norms).astype(np.float32)

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
def arrays_equal(a, b):
    # manual loop with early exit
    for i in range(a.shape[0]):
        if a[i] != b[i]:
            return False
    return True

@nb.njit(fastmath=True)
def recov(helper, vec, scale):
    return round_scaled(helper + vec, scale)

@nb.njit(fastmath=True)
def match(c_vec, q_vec, offset, unit_vectors,
          ring_radius, inner_scale, outer_scale):
    helper, a = gen(c_vec, offset * inner_scale)
    b = recov(helper, q_vec, offset * inner_scale)

    if a[0] == b[0] and arrays_equal(a, b):
        return True

    for i in range(unit_vectors.shape[0]):
        vec = unit_vectors[i]
        ofc = c_vec + vec * ring_radius * offset

        helper, a = gen(ofc, offset * outer_scale)
        b = recov(helper, q_vec, offset * outer_scale)

        if a[0] == b[0] and arrays_equal(a, b):
            return True

    return False

def match_wrapper(args):
    cen, val, rad = args
    return 1 if match(cen, val, rad, UNIT_VECTORS) else 0

@nb.njit(parallel=True, fastmath=True)
def match_batch_full(cen, vals, rad, unit_vectors,
                     ring_radius, inner_scale, outer_scale):
    count = 0
    for i in nb.prange(vals.shape[0]):
        if match(cen, vals[i], rad, unit_vectors,
                 ring_radius, inner_scale, outer_scale):
            count += 1
    return count

def run_bin_search(data, unit_vectors):
    keys = list(data.keys())

    tchk = fchk = 0
    tks = fks = 0

    for key in tqdm(data.keys()):
        cen = data[key][0]
        rad = data[key][1]

        vals_t = np.asarray(data[key][2])
        vals_f = np.asarray(data[key][3])

        tchk += match_batch_full(cen, vals_t, rad, unit_vectors, RING_RADIUS, INNER_LATTICE_SCALE, OUTER_LATTICE_SCALE)
        fchk += match_batch_full(cen, vals_f, rad, unit_vectors, RING_RADIUS, INNER_LATTICE_SCALE, OUTER_LATTICE_SCALE)

        tks += len(vals_t)
        fks += len(vals_f)
        
    return tchk / tks, fchk / fks


def run_sweep(data, save_path, unit_vectors):
    global OUTER_LATTICE_SCALE, RING_RADIUS
    
    res_ma = {"outer": [], "radius":[], "TMR": [], "FMR": []}
   
    scales = [0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6]
    for sc in scales:
        for rad in [0.8, 0.85, 0.9, 0.95, 1.0]:
        
            OUTER_LATTICE_SCALE = sc
            RING_RADIUS = rad
            tmr, fmr = run_bin_search(data, unit_vectors)
    
            res_ma["outer"].append(OUTER_LATTICE_SCALE)
            res_ma["radius"].append(RING_RADIUS)
            res_ma["TMR"].append(tmr)
            res_ma["FMR"].append(fmr)
                
            print(res_ma)

    print("Done! Check:", save_path)
    pd.DataFrame.from_dict(res_ma, orient='columns').to_csv(save_path, index=False)

def main():
    src_file = sys.argv[1]
    dst_file = sys.argv[2]
    data = get_data(src_file)

    unit_vectors = precompute_unit_vectors(
        num_vecs=NUM_POTENTIALS,
        dim=128,
        seed=42
    )

    run_sweep(data, dst_file, unit_vectors)

if __name__ == "__main__":
    main()
