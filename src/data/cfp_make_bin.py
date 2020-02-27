import mxnet as mx
from mxnet import ndarray as nd
import argparse
import pickle
import sys
import os

parser = argparse.ArgumentParser(description='Package CFP images')
# general
parser.add_argument('--data-dir', default='/home/night/Datasets/face/cfp-dataset', help='')
parser.add_argument('--image-size', type=str, default='112,112', help='')
parser.add_argument('--output', default='/home/night/PycharmProjects/face/insightface-CBAM/datasets/cfp_val/cfp.bin', help='path to save.')
args = parser.parse_args()
data_dir = args.data_dir
image_size = [int(x) for x in args.image_size.split(',')]
pairs_end = []
def get_paths():
    pairs = []
    prefix = os.path.join(data_dir,'Protocol/')

    #prefix = "/Split/"
    prefix_F = os.path.join(prefix, "Pair_list_F.txt")
    pairs_F = []
    prefix_P = os.path.join(prefix,"Pair_list_P.txt")
    pairs_P = []
    pairs_end = []
    issame_list = []
    #读pairlist文件
    with open(prefix_F, 'r') as f:
        for line in f.readlines()[0:]:
            pair = line.strip().split()
            pairs_F.append(pair[1])
    print(len(pairs_F))
    with open(prefix_P, 'r') as f:
        for line in f.readlines()[0:]:
            pair = line.strip().split()
            pairs_P.append(pair[1])
    print(len(pairs_P))

    #读pair文件
    prefix = os.path.join(data_dir,"Protocol/Split/FP")
    folders_1 = os.listdir(prefix)
    for folder in folders_1:
        sublist = []
        same_list = []
        pairtxt = os.listdir(os.path.join(prefix, folder))
        for pair in pairtxt:
            img_root_path = os.path.join(prefix, folder, pair)
            with open(img_root_path, 'r') as f:
                for line in f.readlines()[0:]:
                    #print(line)
                    pair1 = line.strip().split(',')
                    #print(pair)
                    pairs_end += (os.path.join(data_dir,'Protocol/',pairs_F[int(pair1[0])-1]),os.path.join(data_dir,'Protocol/',pairs_P[int(pair1[1])-1]))
                    #print(pair)
                    if pair == 'same.txt':
                        #print('ok!')
                        issame_list.append(True)
                    else:
                        issame_list.append(False)
    return pairs_end,issame_list


cfp_paths, issame_list = get_paths()
cfp_bins = []
print(len(cfp_paths))
print(len(issame_list))
print(cfp_paths[0])
print(cfp_paths[1])
print(issame_list[0])
# print(issame_list[1])
#lfw_data = nd.empty((len(lfw_paths), 3, image_size[0], image_size[1]))
print(len(issame_list))
i = 0
for path in cfp_paths:
  with open(path, 'rb') as fin:
    _bin = fin.read()
    cfp_bins.append(_bin)
    #img = mx.image.imdecode(_bin)
    #img = nd.transpose(img, axes=(2, 0, 1))
    #lfw_data[i][:] = img
    i+=1
    if i%1000==0:
      print('loading cfp', i)

with open(args.output, 'wb') as f:
  pickle.dump((cfp_bins, issame_list), f, protocol=pickle.HIGHEST_PROTOCOL)
