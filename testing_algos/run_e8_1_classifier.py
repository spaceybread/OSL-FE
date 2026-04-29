import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
from multiprocessing import Pool, cpu_count

def get_data(npz_file): return np.load(npz_file, allow_pickle=True).item()

def _adjust_parity_int_candidate(z, v):
    z = z.copy()
    if (z.sum() & 1) == 1:
        frac = np.abs(v - z)
        i = np.argmin(frac)
        z[i] += 1 if v[i] > z[i] else -1
    return z
def _adjust_parity_half_candidate(y, v):
    y = y.copy()
    z = (y - 0.5).astype(int)
    if (z.sum() & 1) == 1:
        frac = np.abs(v - y)
        i = np.argmin(frac)
        y[i] += 1.0 if v[i] > y[i] else -1.0
    return y
def round_E8(v):
    v = np.asarray(v, dtype=float)
    if v.shape != (8,):
        raise ValueError("Input must be shape (8,)")
    z0 = np.round(v).astype(int)
    z = _adjust_parity_int_candidate(z0, v)
    d1 = np.sum((v - z)**2)
    y0 = np.round(v - 0.5) + 0.5
    y = _adjust_parity_half_candidate(y0, v)
    d2 = np.sum((v - y)**2)
    return z if d1 <= d2 else y
def round_scaled_E8(v, scale=1.0):
    arr = np.asarray(v, dtype=float)
    scaled = arr / scale
    nearest = round_E8(scaled)
    return nearest * scale
def round_E8_blockwise(v):
    v = np.asarray(v, dtype=float)
    n = v.size
    k = (n + 7) // 8
    padded = np.zeros(8 * k)
    padded[:n] = v
    out = np.zeros_like(padded)
    for i in range(k):
        block = padded[8*i : 8*(i+1)]
        out[8*i : 8*(i+1)] = round_E8(block)
    return out[:n]
def round_scaled(vec, scale):
    v = np.asarray(vec)
    return round_E8_blockwise(v / scale) * scale
def random_vector(n): return np.random.uniform(0, 100, size=n)
def gen(vec, scale):
    rdm = round_scaled(random_vector(len(vec)), scale)
    helper = rdm - vec
    return helper, rdm
def recov(helper, vec, scale):
    return round_scaled(helper + vec, scale)
def match(c_vec, q_vec, scale):
    OFFSET = scale / 2
    helper, a = gen(c_vec, scale)
    b = recov(helper, q_vec, scale)
    if np.array_equal(a, b): return True
    potentials = []
    for i in range(len(c_vec)):
        up = np.copy(c_vec)
        do = np.copy(c_vec)
        up[i] += OFFSET
        do[i] -= OFFSET
        potentials += [up, do]
    n_scale = scale * 0.95
    for x in potentials:
        helper, a = gen(x, n_scale)
        b = recov(helper, q_vec, n_scale)
        if np.array_equal(a, b): return True
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
    coeff = float(sys.argv[3]) * 100
    data = get_data(src_file)
    run_sweep(data, dst_file, coeff)

if __name__ == "__main__": main()
