from sklearn.decomposition import PCA
from collections import defaultdict
import numpy as np
import random

def load_data(npz_file, text_file, save_path, idx, T, R, new_dim):
    npz = np.load(npz_file)
    npz = np.squeeze(npz)
    ids = [x.strip().split('/')[-idx] for x in open(text_file)]

    if (len(ids) != npz.shape[0]): print("Size Mismatch")

    grp = defaultdict(list)
    for s, v in zip(ids, npz): grp[s].append(v)
    
    rem = []

    for k in grp.keys(): 
        if len(grp[k]) < T + R: rem.append(k)
        
    for r in rem: grp.pop(r)
    
    # for each key in grp, grp[k] contains
    # about 18 or so vectors
    
    # print(grp)
    
    all_vecs = np.vstack([np.array(vs) for vs in grp.values()])
    print(all_vecs.shape)
    pca = PCA(n_components=new_dim)
    pca.fit(all_vecs)

    reduced_grp = {}
    for k, vecs in grp.items():
        vecs = np.array(vecs)
        reduced_grp[k] = list(pca.transform(vecs))
    
    seed = 314
    
    keys = list(reduced_grp.keys())
    random.seed(seed)
    random.shuffle(keys)

    split_idx = int(len(keys) * 0.25)
    train_keys = keys[:split_idx]
    test_keys = keys[split_idx:]

    train_data = {k: reduced_grp[k] for k in train_keys}
    test_data = {k: reduced_grp[k] for k in test_keys}

    print(f"Total groups: {len(keys)} | Train: {len(train_data)} | Test: {len(test_data)}")
    
    np.save(save_path + "/train_data.npy", train_data, allow_pickle=True)
    np.save(save_path + "/test_data.npy", test_data, allow_pickle=True)


def make_datasets(npz_file, save_path_rad, save_path_lat, T, R, isTesting):
    data = np.load(npz_file, allow_pickle=True).item()
    all_keys = list(data.keys())
    #print(len(all_keys))

    ma, mb = {}, {}

    for x in data.keys():
        #try: 
        booga_boo = random.sample(data[x], k = T + R)
        #except:
        #    print(len(data[x]), R, T)
        #    continue
        
        registration_vecs = booga_boo[:R]
        intra_vecs = booga_boo[R:]

        inter_vecs = []
        inter_key = x
        
        out_v = T*T if isTesting else T
        
        for i in range(out_v):
            while inter_key == x: inter_key = random.choices(all_keys, k = 1)[0]
            inter_vecs.append(random.choices(data[inter_key], k = 1)[0])

        center = np.mean(registration_vecs, axis = 0)
        init_rad = np.max([np.linalg.norm(ve - center) for ve in registration_vecs])
        
        in_dist = [np.linalg.norm(ve - center) for ve in intra_vecs]
        out_dist = [np.linalg.norm(ve - center) for ve in inter_vecs]

        ma[x] = (init_rad, in_dist, out_dist)
        mb[x] = (center, init_rad, intra_vecs, inter_vecs)

    np.save(save_path_rad, ma)
    np.save(save_path_lat, mb)
    

def split_datasets(rad_file, lat_file, save_train_rad, save_test_rad, save_train_lat, save_test_lat, test_ratio=0.75, seed=314):

    np.random.seed(seed)

    rad_data = np.load(rad_file, allow_pickle=True).item()
    lat_data = np.load(lat_file, allow_pickle=True).item()

    common_keys = list(set(rad_data.keys()) & set(lat_data.keys()))
    if len(common_keys) == 0:
        print("No overlapping keys between radius and latent data files.")
        return

    np.random.shuffle(common_keys)
    split_idx = int(len(common_keys) * (1 - test_ratio))
    train_keys = common_keys[:split_idx]
    test_keys = common_keys[split_idx:]

    train_rad = {k: rad_data[k] for k in train_keys}
    test_rad = {k: rad_data[k] for k in test_keys}

    train_lat = {k: lat_data[k] for k in train_keys}
    test_lat = {k: lat_data[k] for k in test_keys}

    np.save(save_train_rad, train_rad)
    np.save(save_test_rad, test_rad)
    np.save(save_train_lat, train_lat)
    np.save(save_test_lat, test_lat)

    print(f"Split complete:")
    print(f"  Training set: {len(train_keys)} samples")
    print(f"  Testing set:  {len(test_keys)} samples")

    
