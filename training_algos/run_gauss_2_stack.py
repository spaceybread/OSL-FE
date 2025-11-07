import pandas as pd
import numpy as np
from tqdm import tqdm
import sys

def get_data(npz_file): return np.load(npz_file, allow_pickle=True).item()

def round_scaled(vec, scale): return np.round(vec / scale) * scale

def random_vector(n): return np.random.uniform(0, 100, size=n)

def gen(vec, scale):
    rdm = round_scaled(random_vector(len(vec)), scale)
    helper = rdm - vec
    return helper, rdm

def recov(helper, vec, scale):
    return round_scaled(helper + vec, scale)

from numba import njit

@njit(fastmath=True, cache=True)
def match(c_vec, q_vec, scale):
    OFFSET = scale / 2
    n = len(c_vec)
    n0_scale = scale * 1.0
    n1_scale = scale * 1.0

    # Precompute randoms
    rand_base = np.random.uniform(0, 100, n)

    # Base check
    rdm = np.round(rand_base / scale) * scale
    helper = rdm - c_vec
    b = np.round((helper + q_vec) / scale) * scale
    if np.all(rdm == b):
        return True

    # Single offset checks
    for i in range(n):
        for sign in (+1, -1):
            x = c_vec.copy()
            x[i] += sign * OFFSET
            rdm = np.round((np.random.uniform(0, 100, n)) / n0_scale) * n0_scale
            helper = rdm - x
            b = np.round((helper + q_vec) / n0_scale) * n0_scale
            if np.all(rdm == b):
                return True

    # Pairwise
    for i in range(n - 1):
        for j in range(i + 1, n):
            for s0 in (+1, -1):
                for s1 in (+1, -1):
                    x = c_vec.copy()
                    x[i] += s0 * OFFSET
                    x[j] += s1 * OFFSET
                    rdm = np.round((np.random.uniform(0, 100, n)) / n1_scale) * n1_scale
                    helper = rdm - x
                    b = np.round((helper + q_vec) / n1_scale) * n1_scale
                    if np.all(rdm == b):
                        return True
    return False


def run_bin_search(data, alpha): 
    hi, lo = 1, 0

    keys = list(data.keys())
    res = {}

    for _ in tqdm(range(16)): 
        tchk, fchk = 0, 0
        tks, fks = 0, 0
        
        coeff = (hi + lo) / 2

        for key in tqdm(keys):
            rad = data[key][1] * coeff
            cen = data[key][0]

            tchk += sum([1 if match(cen, val, rad) else 0 for val in data[key][2]])
            tks += len(data[key][2])
            fchk += sum([1 if match(cen, val, rad) else 0 for val in data[key][3]])
            fks += len(data[key][3])

        tmr, fmr = tchk / tks, fchk / fks
        res[coeff] = [tmr, fmr]

        if tmr > alpha: hi = coeff
        else: lo = coeff

    return res, coeff

def run_sweep(data, save_path): 
    res_ma = {"coeff": [], "TMR": [], "FMR": []}

    for i in [70, 90]:
        resdb, idx = run_bin_search(data, i / 100)
        
        res_ma["coeff"].append(idx)
        res_ma["TMR"].append(resdb[idx][0])
        res_ma["FMR"].append(resdb[idx][1])
    
    print("Done! Check:", save_path)
    pd.DataFrame.from_dict(res_ma, orient='columns').to_csv(save_path, index=False)

def main(): 
    src_file = sys.argv[1]
    dst_file = sys.argv[2]

    data = get_data(src_file)
    run_sweep(data, dst_file)

if __name__ == "__main__": main()
