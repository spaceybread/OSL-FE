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
    

def run_bin_search(data, alpha): 
    hi, lo = 1, 0

    keys = list(data.keys())
    res = {}

    for _ in tqdm(range(20)): 
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

main()
