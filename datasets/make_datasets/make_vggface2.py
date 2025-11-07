import make_smaller

data_dir = "/Users/home/Documents/OSL-FE/datasets/vggface2_128/"
src_dir = "/Users/home/Documents/curly-waffle/face-data/vggface2-resnet/" 

npz_file = src_dir + "embeddings.npy"
text_file = src_dir + "files.txt"

T, R = 10, 8

mapped_data = data_dir + "/inter/"
train_data = data_dir + "/inter/" + "train_data.npy"
tests_data = data_dir + "/inter/" + "test_data.npy"

train_rad = data_dir + "/train/" + "radial_ds.npy"
train_lat = data_dir + "/train/" + "lat_ds.npy"
tests_rad = data_dir + "/tests/" + "radial_ds.npy"
tests_lat = data_dir + "/tests/" + "lat_ds.npy"

make_smaller.load_data(npz_file, text_file, mapped_data, 2, T, R, 128)

make_smaller.make_datasets(train_data, train_rad, train_lat, T, R, False)
make_smaller.make_datasets(tests_data, tests_rad, tests_lat, T, R, True)
