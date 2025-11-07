import pandas as pd
import numpy as np
from tqdm import tqdm
import sys

def get_data(npz_file): return np.load(npz_file, allow_pickle=True).item()

def run_bin_search(data, coeff): 

    keys = list(data.keys())


    tchk, fchk = 0, 0
    tks, fks = 0, 0
        
    for key in keys:
        rad = data[key][0] * coeff
            
        tchk += sum([1 if val < rad else 0 for val in data[key][1]])
        tks += len(data[key][1])
        fchk += sum([1 if val < rad else 0 for val in data[key][2]])
        fks += len(data[key][2])

    tmr, fmr = tchk / tks, fchk / fks
    return tmr, fmr

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
    coeff = float(sys.argv[3])

    data = get_data(src_file)
    run_sweep(data, dst_file, coeff)

main()
