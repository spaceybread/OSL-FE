import numpy as np
import sys

data = np.load(sys.argv[1], allow_pickle=True).item()

print(len(list(data.keys())))
k = list(data.keys())[0]
print(len(data[k][2][0]))
