import numpy as np
import h5py, os, cv2, random, json, shutil, logging


def split_data(config, override=False):
    """
    split 3D CT data to 2D slices
    :config: instance of class 'Configuration'
    :override: (bool) override the existed folder if set to True
    """
    img_dir = config.data_root_path
    logging.info(img_dir)
    if override:
        if os.path.exists(img_dir):
            shutil.rmtree(img_dir)
    else:
        assert not os.path.exists(img_dir), "target folder exists"

    
    split_dirs = {x: os.path.join(img_dir, x) for x in ['train', 'test', 'valid']}
    for k,v in split_dirs.items():
        os.makedirs(v)


    counter = 0

    for ind in range(1, 11):
        logging.info("File {}".format(ind))

        fp = h5py.File(os.path.join(config.h5data_path, "case{}.h5".format(ind)), 'r')

        r = np.array(fp['data']).astype(np.uint8)
        #r = np.stack([r], axis=3)
        l = np.array(fp['annot']).astype(np.uint8)


        if config.name == "axis1":
            r = r[r.shape[0]%16: ,...]
            l = l[l.shape[0]%16: ,...]
            r = np.transpose(r, (1,0,2,3))
            l = np.transpose(l, (1,0,2))
        elif config.name == 'axis2':
            r = r[r.shape[0]%16: ,...]
            l = l[l.shape[0]%16: ,...]
            r = np.transpose(r, (2,0,1,3))
            l = np.transpose(l, (2,0,1))



        keep = l.sum(axis=(1,2)) > 0


        if ind==10:
            folder = "test"
        else:
            folder = "train"


        for i in range(r.shape[0]):
            with h5py.File(os.path.join(split_dirs[folder], "{:05d}.h5".format(counter)), "w") as f:
                f['data']  = r[i, ...]
                f['annot'] = l[i, ...]

            counter += 1


    flist = [x.split('.')[0] for x in os.listdir(split_dirs['train']) if x.split('.')[-1]=='h5']
    random.shuffle(flist)
    n_valid = int(len(flist)*0.1)

    for fname in flist[:n_valid]:
        source = os.path.join(split_dirs['train'], fname+'.h5')
        target = os.path.join(split_dirs['valid'], fname+'.h5')
        shutil.move(source, target)


    for k,v in split_dirs.items():
        logging.info("{}: {} slices".format(k, len(os.listdir(v))))




def nii_to_hdf5(DATA_PATH = "nii_label", OUT_PATH  = "h5_8b"):
    """
    convert .nii files to hdf5 files
    :DATA_PATH: input .nii folder. The folder must contain 'case?.nii.gz' and 'case?_label.nii.gz'
    :OUT_PATH: output .h5 folder
    *requirement: 'SimpleITK' package (pip install SimpleITK)
    """
    import SimpleITK as sitk

    if not os.path.exists(OUT_PATH):
        os.mkdir(OUT_PATH)
        
    for ind in range(1, 13):
        
        path = os.path.join(DATA_PATH, 'case{}.nii.gz'.format(ind))
        img = sitk.ReadImage(path)
        data = sitk.GetArrayFromImage(img)  # 0~4095 (12bit gray)
        
        print("Shape:", data.shape)
        
        if ind in [11,12]:  data = data+1024
        data = (data/16).astype(np.uint8)
        
        path = os.path.join(DATA_PATH, 'case{}_label.nii.gz'.format(ind))
        img = sitk.ReadImage(path)
        annot = sitk.GetArrayFromImage(img)
        
        f = h5py.File(os.path.join(OUT_PATH, 'case{}.h5'.format(ind)), 'w')
        f['data'] = data
        f['annot'] = annot
        f.close()







# 1 (559, 512, 512), 0-18
# 2 (507, 512, 512), 0-17
# 3 (560, 512, 512), 0-17
# 4 (625, 512, 512), 0-19
# 5 (601, 512, 512), 0-18
# 6 (562, 512, 512), 0-17
# 7 (509, 512, 512), 0-17
# 8 (548, 512, 512), 0-18
# 9 (572, 512, 512), 0-18
# 10 (552, 512, 512), 0-17