import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
from multiprocessing import Pool, cpu_count

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

    rand_base = np.random.uniform(0, 100, n)

    rdm = np.round(rand_base / scale) * scale
    helper = rdm - c_vec
    b = np.round((helper + q_vec) / scale) * scale
    if np.all(rdm == b):
        return True

    for i in range(n):
        for sign in (+1, -1):
            x = c_vec.copy()
            x[i] += sign * OFFSET
            rdm = np.round((np.random.uniform(0, 100, n)) / n0_scale) * n0_scale
            helper = rdm - x
            b = np.round((helper + q_vec) / n0_scale) * n0_scale
            if np.all(rdm == b):
                return True

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


_data = None
_coeff = None

def _init(data, coeff):
    global _data, _coeff
    _data = data
    _coeff = coeff

def process_key(key):
    rad = _coeff
    cen = _data[key][0]
    tchk = sum(1 if match(cen, val, rad) else 0 for val in _data[key][2])
    tks  = len(_data[key][2])
    fchk = sum(1 if match(cen, val, rad) else 0 for val in _data[key][3])
    fks  = len(_data[key][3])
    return tchk, tks, fchk, fks


def run_bin_search(data, coeff):
    keys = list(data.keys())

    with Pool(
        processes=cpu_count(),
        initializer=_init,
        initargs=(data, coeff),
    ) as pool:
        results = list(tqdm(pool.imap(process_key, keys), total=len(keys)))

    tchk = sum(r[0] for r in results)
    tks  = sum(r[1] for r in results)
    fchk = sum(r[2] for r in results)
    fks  = sum(r[3] for r in results)

    return tchk / tks, fchk / fks


def run_sweep(data, save_path, COEFF):
    res_ma = {"coeff": [], "TMR": [], "FMR": []}

    tmr, fmr = run_bin_search(data, COEFF)

    res_ma["coeff"].append(COEFF)
    res_ma["TMR"].append(tmr)
    res_ma["FMR"].append(fmr)

    print("Done! Check:", save_path)
    pd.DataFrame.from_dict(res_ma, orient='columns').to_csv(save_path, index=False)

def main():
    src_file = sys.argv[1]
    dst_file = sys.argv[2]
    coeff    = float(sys.argv[3]) * 100

    data = get_data(src_file)
    run_sweep(data, dst_file, coeff)

if __name__ == "__main__": main()
