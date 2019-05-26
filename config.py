class Configuration:
    def __init__(self, exp_name):
        assert exp_name in ['axis0', 'axis1', 'axis2']
        self.name               = exp_name
        
        # training hyper-parameters
        self.img_ch             = 1
        self.output_ch          = 2
        self.num_epochs         = 10
        self.num_epochs_decay   = 100
        
        self.batch_size         = 1
        self.valid_batch_size   = 1
        self.test_batch_size    = 1
        
        self.num_workers        = 8
        self.lr                 = 5e-4
        self.beta1              = 0.5         # momentum1 in Adam
        self.beta2              = 0.999       # momentum2 in Adam

        self.log_step           = 1
        self.val_step           = 1

        # misc
        self.mode               = 'train'
        self.model_type         = 'U_Net'     # 'U_Net/R2U_Net/AttU_Net/R2AttU_Net'
        self.data_mode          = 'raw'       # raw/location
        
        self.model_path         = '/content/drive/models/{}'.format(exp_name)
        self.h5data_path        = '/content/drive/h5_8b'
        self.data_root_path     = '/content/data/{}/'.format(exp_name)
        #self.train_path         = '/content/data/{}/train/'.format(exp_name)
        #self.valid_path         = '/content/data/{}/valid/'.format(exp_name)
        #self.test_path          = '/content/data/{}/test/'.format(exp_name)
        self.result_path        = '/content/drive/log/{}'.format(exp_name)    # LOG_DIR

        self.cuda_idx           = 1
    
    
    def __str__(self):
        return "name={}, n_epoch={}, batch_size={}, lr={}, mode={}".format(
            self.name, self.num_epochs, self.batch_size, self.lr, self.mode)

    def check_constraint(self):
        assert self.exp_name in ['axis0', 'axis1', 'axis2']
        assert self.output_ch in [2, 20]
        assert self.mode in ['train', 'valid', 'test']
        assert self.model_type in ['U_Net']
        assert self.data_mode in ['raw', 'location']
        if self.data_mode=='location':
            assert self.img_ch==3
        elif self.data_mode=='raw':
            assert self.img_ch==1


