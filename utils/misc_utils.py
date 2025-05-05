import json
import numpy as np
import seaborn as sns

from scipy.spatial.transform import Rotation


ShapeNetIDMap = {'4379243': 'table', '3593526': 'jar', '4225987': 'skateboard', '2958343': 'car', '2876657': 'bottle', 
                 '4460130': 'tower', '3001627': 'chair', '2871439': 'bookshelf', '2942699': 'camera', '2691156': 'airplane', 
                 '3642806': 'laptop', '2801938': 'basket', '4256520': 'sofa', '3624134': 'knife', '2946921': 'can', 
                 '4090263': 'rifle', '4468005': 'train', '3938244': 'pillow', '3636649': 'lamp', '2747177': 'trash_bin', 
                 '3710193': 'mailbox', '4530566': 'watercraft', '3790512': 'motorbike', '3207941': 'dishwasher', 
                 '2828884': 'bench', '3948459': 'pistol', '4099429': 'rocket', '3691459': 'loudspeaker', 
                 '3337140': 'file cabinet', '2773838': 'bag', '2933112': 'cabinet', '2818832': 'bed', 
                 '2843684': 'birdhouse', '3211117': 'display', '3928116': 'piano', '3261776': 'earphone', 
                 '4401088': 'telephone', '4330267': 'stove', '3759954': 'microphone', '2924116': 'bus', '3797390': 'mug', 
                 '4074963': 'remote', '2808440': 'bathtub', '2880940': 'bowl', '3085013': 'keyboard', '3467517': 'guitar', 
                 '4554684': 'washer', '2834778': 'bicycle', '3325088': 'faucet', '4004475': 'printer', '2954340': 'cap', 
                 '3046257': 'clock', '3513137': 'helmet', '3991062': 'flowerpot', '3761084': 'microwaves'}
ShapeNetLabels = ['void',
                  'table', 'jar', 'skateboard', 'car', 'bottle',
                  'tower', 'chair', 'bookshelf', 'camera', 'airplane',
                  'laptop', 'basket', 'sofa', 'knife', 'can',
                  'rifle', 'train', 'pillow', 'lamp', 'trash_bin',
                  'mailbox', 'watercraft', 'motorbike', 'dishwasher', 'bench',
                  'pistol', 'rocket', 'loudspeaker', 'file cabinet', 'bag',
                  'cabinet', 'bed', 'birdhouse', 'display', 'piano',
                  'earphone', 'telephone', 'stove', 'microphone', 'bus',
                  'mug', 'remote', 'bathtub', 'bowl', 'keyboard',
                  'guitar', 'washer', 'bicycle', 'faucet', 'printer',
                  'cap', 'clock', 'helmet', 'flowerpot', 'microwaves']
ScanNet_OBJ_CLASS_IDS = np.array([1, 2, 3, 5, 6, 7, 8, 11, 12, 13, 15,
                                  18, 19, 20, 21, 23, 24, 25, 28, 29, 30, 
                                  31, 32, 34, 35, 37, 38, 39, 41, 42, 43, 
                                  44, 45, 46, 47, 48, 49, 50, 51,
                                  52, 54, 55])
CAD_labels = ['table', 'jar', 'bottle', 'chair', 'bookshelf',
              'laptop', 'basket', 'sofa', 'can', 'pillow', 
              'lamp', 'trash_bin', 'motorbike', 'dishwasher', 'bench', 
              'loudspeaker', 'file cabinet', 'bag', 'cabinet', 'bed', 
              'display', 'piano', 'stove', 'bathtub', 'bowl', 
              'keyboard', 'guitar', 'washer', 'faucet', 'printer',
              'cap', 'clock', 'flowerpot', 'microwaves']

palette_cls = (255 * np.array([(0., 0., 0.), *sns.color_palette("hls", len(ShapeNetLabels))])).astype(np.uint8)


def fnv_hash_vec(arr):
    """
    FNV64-1A
    """
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * np.ones(arr.shape[0], dtype=np.uint64)
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr


def ravel_hash_vec(arr):
    """
    Ravel the coordinates after subtracting the min coordinates.
    """
    assert arr.ndim == 2
    arr = arr.copy()
    arr -= arr.min(0)
    arr = arr.astype(np.uint64, copy=False)
    arr_max = arr.max(0).astype(np.uint64) + 1

    keys = np.zeros(arr.shape[0], dtype=np.uint64)
    # Fortran style indexing
    for j in range(arr.shape[1] - 1):
        keys += arr[:, j]
        keys *= arr_max[j + 1]
    keys += arr[:, -1]
    return keys


def rotation_matrix(rot_x, rot_y, rot_z):
    """
    Construct a rotation matrix.
    """
    Rx = np.array([[1., 0., 0.],
                   [0., np.cos(rot_x), -np.sin(rot_x)],
                   [0., np.sin(rot_x), np.cos(rot_x)]])
    Ry = np.array([[np.cos(rot_y), 0., np.sin(rot_y)],
                   [0., 1., 0.],
                   [-np.sin(rot_y), 0., np.cos(rot_y)]])
    Rz = np.array([[np.cos(rot_z), -np.sin(rot_z), 0.],
                   [np.sin(rot_z), np.cos(rot_z), 0.],
                   [0., 0., 1.]])
    R = np.dot(Rz, np.dot(Ry, Rx))

    return R


def random_rotation():
    """
    Generate a random rotation matrix.
    """
    rot_x = 0. #np.random.uniform(low=0.0, high=2 * np.pi)
    rot_y = np.random.uniform(low=0.0, high=2 * np.pi)
    rot_z = 0. #np.random.uniform(low=0.0, high=2 * np.pi)

    R = rotation_matrix(rot_x, rot_y, rot_z)

    return R


def random_translation(min=-0.25, max=0.25):
    """
    Generate a random translation vector between [min, max].
    """
    t = np.random.uniform(low=min, high=max, size=(1, 3))

    return t


def random_scale(min=0.9, max=1.1):
    """
    Generate a random scaling factor between [min, max].
    """
    s = np.random.uniform(low=min, high=max)

    return s


def read_txt(file):
    with open(file, 'r') as f:
        output = [x.strip() for x in f.readlines()]
    return output


def read_json(filename):
    with open(filename, 'r') as infile:
        return json.load(infile)
    

def make_M_from_tqs(t, q, s):
    q = np.roll(np.array(q), -1)
    r = Rotation.from_quat(q)
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = r.as_matrix()
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)

    M = T.dot(R).dot(S)

    return M 


def Rx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])


def transform_to_ScanNet(negate=True):
    deg = 90 if not negate else -90
    rot_tmp1 = Rx(np.deg2rad(deg))
    T = np.eye(4)
    T [:3, :3] = rot_tmp1
    return T


def extract_label(f):
    clsname = f[:-4].split('_')[-2]
    if clsname == 'trash': clsname = 'trash_bin'
    if clsname == 'bin': clsname = 'trash_bin'
    if clsname not in CAD_labels:
        return None
    else:
        return CAD_labels.index(clsname)


def extract_score(f):
    return float(f[:-4].split('_')[-1])