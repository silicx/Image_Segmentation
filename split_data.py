import numpy as np
import h5py, os, cv2, random, json, shutil

def split_data(config):
    img_dir = os.path.join(config.data_root_path, config.name)
    logging.info(img_dir)
    assert os.path.exists(img_dir), "target folder exists"

    
    split_dirs = {x: os.path.join(img_dir, x) for x in ['train', 'test', 'valid']}
    for k,v in split_dirs.items():
        os.makedirs(v)


    counter = 0

    for ind in range(1, 11):
        print(ind)

        fp = h5py.File(os.path.join(config.h5data_path, "case{}.h5".format(ind)), 'r')

        r = np.array(fp['raw'])
        r = np.stack([r,r,r], axis=3).astype(np.uint8)
        l = np.array(fp['label']).astype(np.uint8)


        if config.exp_name == "axis1":
            r = r[r.shape[0]%16: ,...]
            l = l[l.shape[0]%16: ,...]
            r = np.transpose(r, (1,0,2,3))
            l = np.transpose(l, (1,0,2))
        elif config.exp_name == 'axis2':
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
        print(k, len(os.listdir(v)))