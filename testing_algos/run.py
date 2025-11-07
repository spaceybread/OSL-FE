# testing algos

from pathlib import Path
import pandas as pd
import glob
import subprocess
import sys
import os

data_dir = sys.argv[1]

lat_ds = data_dir + "/tests/lat_ds.npy"
rad_ds = data_dir + "/tests/radial_ds.npy"

results_folder = data_dir + "/tests/results"

try: os.mkdir(results_folder)
except FileExistsError: pass

files = glob.glob(data_dir + "/train/results/*.csv")

coeffs = {}

for f in files:
    df = pd.read_csv(f)
    coeff_values = df['coeff'].tolist()
    coeffs[f.split('/')[-1].split('.')[0]] = coeff_values

venv_python = Path("~/Desktop/.venv/bin/python3").expanduser()

# radial classifier 70
script_path = "run_radial_classifier.py"
args = [rad_ds, results_folder + "/radial_70.csv", str(min(coeffs['radial']))]
cmd = [str(venv_python), str(script_path)] + args
result = subprocess.run(cmd)

# radial classifier 90
script_path = "run_radial_classifier.py"
args = [rad_ds, results_folder + "/radial_90.csv", str(max(coeffs['radial']))]
cmd = [str(venv_python), str(script_path)] + args
result = subprocess.run(cmd)
    
# gauss no stack classifier 70
script_path = "run_gauss_no_stack.py"
args = [lat_ds, results_folder + "/gauss_0_70.csv", str(min(coeffs['gauss_0']))]
cmd = [str(venv_python), str(script_path)] + args
result = subprocess.run(cmd)

# gauss no stack classifier 90
script_path = "run_gauss_no_stack.py"
args = [lat_ds, results_folder + "/gauss_0_90.csv", str(max(coeffs['gauss_0']))]
cmd = [str(venv_python), str(script_path)] + args
result = subprocess.run(cmd)

# gauss 1 stack classifier 70
script_path = "run_gauss_1_stack.py"
args = [lat_ds, results_folder + "/gauss_1_70.csv", str(min(coeffs['gauss_1']))]
cmd = [str(venv_python), str(script_path)] + args
result = subprocess.run(cmd)

# gauss 1 stack classifier 90
script_path = "run_gauss_1_stack.py"
args = [lat_ds, results_folder + "/gauss_1_90.csv", str(max(coeffs['gauss_1']))]
cmd = [str(venv_python), str(script_path)] + args
result = subprocess.run(cmd)

# gauss 2 stack classifier 70
script_path = "run_gauss_2_stack.py"
args = [lat_ds, results_folder + "/gauss_2_70.csv", str(min(coeffs['gauss_2']))]
cmd = [str(venv_python), str(script_path)] + args
result = subprocess.run(cmd)

# gauss 2 stack classifier 90
script_path = "run_gauss_2_stack.py"
args = [lat_ds, results_folder + "/gauss_2_90.csv", str(max(coeffs['gauss_2']))]
cmd = [str(venv_python), str(script_path)] + args
result = subprocess.run(cmd)

# e8 classifier 70
script_path = "run_e8_classifier.py"
args = [lat_ds, results_folder + "/e8_70.csv", str(min(coeffs['e8']))]
cmd = [str(venv_python), str(script_path)] + args
result = subprocess.run(cmd)

# e8 classifier 90
script_path = "run_e8_classifier.py"
args = [lat_ds, results_folder + "/e8_90.csv", str(max(coeffs['e8']))]
cmd = [str(venv_python), str(script_path)] + args
result = subprocess.run(cmd)
