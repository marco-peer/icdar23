class Config:
    log_dir = 'logs'

    def icdar2017(self):
        trainset = {
            "dataset": "icdar2017",
            "set": "train"
        }
        testset = {
            "dataset": "icdar2017",
            "set": "test"
        }

        return {
            "log_dir": "experiments",
            'logger' : 'wandb',
            "trainset": trainset,
            "testset" : testset,
            'train_label' : 'cluster',

            # only use when train on color
            'grayscale' : False,

            'data_augmentation' : 'morph',

            # base lr 5e-4 to 1e-5
            'optimizer_options': {'optimizer': 'adam', 'base_lr': 1e-4, 'wd': 0, 'final_lr': 1e-5, 'warmup_epochs' : 5},

            "super_fancy_new_name": "test_repository", 

            'model' : {
                'name' : 'resnet56',
                'num_clusters' : 100,
                'encoding' : 'netrvlad',
            },

            'train_options': {'epochs': 30, 
                              'batch_size': 1024,                               
                              'callback' : 'early_stopping',
                              'callback_patience' : 5,

                              'loss' : 'triplet',  
                              'margin' : 0.1,
                              'sampler_m' : 16,
                              'length_before_new_iter': 512000,
            },

            'eval_options' : {
                'pca_dim' : 400,
                'gmp_alpha' : 100
            },

            'test_batch_size': 512
        }

    def icdar2019(self):
        trainset = {
            "dataset": "icdar2019",
            "set": "train"
        }
        testset = {
            "dataset": "icdar2019",
            "set": "test"
        }

        return {
            "log_dir": "experiments",
            'logger' : 'wandb',
            "trainset": trainset,
            "testset" : testset,
            'train_label' : 'cluster',

            'data_augmentation' : 'morph',
            
            # only use when train on color
            'grayscale' : False,

            # base lr 5e-4 to 1e-5
            'optimizer_options': {'optimizer': 'adam', 'base_lr': 5e-4, 'wd': 1e-4, 'final_lr': 1e-5, 'warmup_epochs' : 2},

            "super_fancy_new_name": "netvlad_100_resnet56_icdar2019",

            'model' : {
                'name' : 'resnet56',
                'num_clusters' : 100,
                'encoding' : 'netrvlad'
            },

            'train_options': {'epochs': 25, 
                              'batch_size': 1024,                               
                              'callback' : 'early_stopping',
                              'callback_patience' : 5,

                              'loss' : 'triplet',  
                              'margin' : 0.1,
                              'sampler_m' : 16,
                              'length_before_new_iter': 512000

            },

            'eval_options' : {
                'pca_dim' : 400,
                'gmp_alpha' : 100
            },

            'test_batch_size': 256
        }
    
config__ = [Config().icdar2019]