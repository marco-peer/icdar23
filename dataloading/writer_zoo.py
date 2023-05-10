import os.path
from .regex import ImageFolder

class WriterZoo:

    @staticmethod
    def new(desc, **kwargs):
        return ImageFolder(desc['path'], regex=desc['regex'], **kwargs)

    @staticmethod
    def get(dataset, set, **kwargs):
        _all = WriterZoo.datasets
        d = _all[dataset]
        s = d['set'][set]

        s['path'] = os.path.join(d['basepath'], s['path'])
        return WriterZoo.new(s, **kwargs)

    datasets = {

        'icdar2017': {
            'basepath': '/data/mpeer/resources',
            'set': {
                'test' :  {'path': 'icdar2017_test_sift_patches_binarized',
                                  'regex' : {'writer': '(\d+)', 'page': '\d+-IMG_MAX_(\d+)'}},

                'train' :  {'path': '/icdar2017_train_sift_patches_binarized',
                                  'regex' : {'cluster' : '(\d+)', 'writer': '\d+_(\d+)', 'page' : '\d+_\d+-\d+-IMG_MAX_(\d+)'}},
                
            }
        },

        'icdar2013': {
            'basepath': '/data/mpeer/resources',
            'set': {
                'test' :  {'path': 'icdar2013_test_sift_patches_binarized',
                                  'regex' : {'writer': '(\d+)', 'page': '\d+_(\d+)'}},

                'train' :  {'path': 'icdar2013_train_sift_patches_1000/',
                                  'regex' : {'cluster' : '(\d+)', 'writer': '\d+_(\d+)', 'page' : '\d+_\d+_(\d+)'}}                             
            }
        },
        
        'icdar2019': {
            'basepath': '/data/mpeer/resources',
            'set': {
                'test' :  {'path': 'wi_comp_19_test_patches',
                                  'regex' : {'writer': '(\d+)', 'page': '\d+_(\d+)'}},

                'train' :  {'path': 'wi_comp_19_validation_patches',
                                  'regex' : {'cluster' : '(\d+)', 'writer': '\d+_(\d+)', 'page' : '\d+_\d+_(\d+)'}},
            }
        }
    }