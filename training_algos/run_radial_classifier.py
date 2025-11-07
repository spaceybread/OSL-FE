import pandas as pd
import numpy as np
from tqdm import tqdm
import sys

def get_data(npz_file): return np.load(npz_file, allow_pickle=True).item()

def run_bin_search(data, alpha): 
    hi, lo = 2, 0

    keys = list(data.keys())
    res = {}

    for _ in range(64): 
        tchk, fchk = 0, 0
        tks, fks = 0, 0
        
        coeff = (hi + lo) / 2

        for key in keys: 
            rad = data[key][0] * coeff
            
            tchk += sum([1 if val < rad else 0 for val in data[key][1]])
            tks += len(data[key][1])
            fchk += sum([1 if val < rad else 0 for val in data[key][2]])
            fks += len(data[key][2])

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
