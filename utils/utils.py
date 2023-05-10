import logging
from copy import deepcopy

import time
import os
import shutil
import json
import datetime
import yaml
from tensorboardX import SummaryWriter
import torch
import collections
import numpy as np
import torch
import random
import pickle
import socket
import git
import matplotlib as mpl
import wandb

from torchvision import datasets, transforms
from PIL import Image
from PIL import ImageFile
from PIL import ImageOps, Image, ImageDraw, ImageFont

mpl.use('Agg')
import matplotlib.pyplot as plt


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def save_model(model, optimizer, epoch, path):
    # save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def getLogger(name):
    if name.lower() == 'tensorboard':
        return TensorboardLogger
    if name.lower() == 'wandb':
        return WandbLogger
    logging.info("Using default logger: Tensorboard")
    return TensorboardLogger

class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True

def save_yaml(dir, argdict):
    with open(dir, 'w') as file: 
        documents = yaml.dump(argdict, file, Dumper=NoAliasDumper)

def load_yaml(path):
    with open(path) as fin:
        argdict = yaml.safe_load(fin)
    return argdict

def decorate(v, d, label, color='grey', height=None):
    height = 1024 if not height else height
    fnt = ImageFont.truetype('utils/font/Montserrat-Regular.otf', size=int(height/10))   
    tf = transforms.Compose([
               transforms.Resize((height, height * 4)),
               transforms.ToPILImage() 
    ])
    image = tf(v) #norm(v))
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), "{:.4f}".format(d), fill=color,
              font=fnt)
    draw.text((10, 190), label, fill=color,
              font=fnt)
    return ImageOps.expand(image=image, border=10, fill=color)

class AbstractLogger(object):
    def __init__(self, log_dir, modules=None, args=None):
        self.log_dir = self._prepare_log_dir(log_dir, modules)

        self.args = args
        self.config= os.path.join(self.log_dir, 'config.yaml')
        save_yaml(self.config, self.args)
        logging.info("writing config file to: %s ", self.config)

        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.global_step = 0
        self.epoch = None
    
    def step(self, step=1):
        self.global_step += step


    @staticmethod
    def _prepare_log_dir(log_path, save_modules=None):
        if save_modules is None:
            save_modules = ['__main__']
        else:
            save_modules = ['__main__'] + save_modules

        import datetime
        now = datetime.datetime.now()
        log_path = log_path + '-%d-%02d-%02d-%02d-%02d' % (now.year, now.month, now.day, now.hour, now.minute)
        if os.path.isdir(log_path):
            log_path += '-%02d' % now.second

        os.mkdir(os.path.expanduser(log_path))

        file_handler = logging.FileHandler(os.path.join(log_path, 'log.txt'), mode='w')
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s '))
        logging.info("writing log file to: %s ", os.path.join(log_path, 'log.txt'))
        logging.getLogger().addHandler(file_handler)

        import sys
        for module in save_modules:
            shutil.copy(os.path.abspath(sys.modules[module].__file__), log_path)
        return log_path


    def log_options(self, options, changes=None):
        if type(options) != dict:
            options = options.__dict__

        options['hash'] = json_hash(options)

        with open(os.path.join(self.log_dir, 'options.json'), 'w') as fp:
            json.dump(options, fp, indent=4)

        if changes:
            with open(os.path.join(self.log_dir, 'changes.json'), 'w') as fp:
                json.dump(changes, fp, indent=4)

    def log_dict(self, dict_to_log, prefix=None, suffix=None, stdout=True, step=None):
        for k, v in dict_to_log.items():
            name = '-'.join(filter(None, [prefix, k, suffix]))
            self.log_value(name, v, step)
            if stdout:
                logging.info('{} {:5f}'.format(name, v))

class TensorboardLogger(AbstractLogger):
    

    def log_value(self, name, value, step=None):
        if not step:
            step = self.global_step
        self.writer.add_scalar(name, value, step)

        return self

    def add_figure(self, tag, figure, close=True, walltime=None):
        self.writer.add_figure(tag, figure, global_step=self.global_step, close=close, walltime=walltime)
        return self

    def log_embedding(self, features, labels, labels_header=None, images=None, step=None, name='default'):
        if not step:
            step = self.global_step

        if images is not None:
            images = torch.Tensor(images)

            for k, img in enumerate(images):
                img = (img - img.min()) / (img.max() - img.min())
                images[k] = img

        self.writer.add_embedding(torch.Tensor(features), labels, images, step, tag=name, metadata_header=labels_header)
        return self



    def add_pr_curve_from_dict_list(self, dict_list, step=None, name='ROC'):
        if not step and self.epoch:
            suffix = self.epoch
        elif not step:
            suffix = self.global_step
        else:
            suffix = step

        with open(os.path.join(self.log_dir, 'pr-curve-{}.pkl'.format(suffix)), 'wb') as fp:
            pickle.dump(dict_list, fp)

        true_positive_counts = [d['true_positives'] for d in dict_list]
        false_positive_counts = [d['false_positives'] for d in dict_list]
        true_negative_counts = [d['true_negatives'] for d in dict_list]
        false_negative_counts = [d['false_negatives'] for d in dict_list]
        precision = [d['precision'] for d in dict_list]
        recall = [d['recall'] for d in dict_list]
        thresh = [d['threshold'] for d in dict_list]

        fig, ax1 = plt.subplots()
        ax1.plot(recall, '-r', label='recall')
        ax1.plot(precision, '-b', label='precision')
        ax1.set_ylabel('precision', color='b')
        ax1.legend()

        ax2 = ax1.twinx()
        ax2.plot(thresh, '-g', label='threshold')
        ax2.set_ylabel('threshold')
        ax2.legend()
        fig.tight_layout()

        #    fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(18.5, 10.5)
        fig.savefig(os.path.join(self.log_dir, 'pr-curve-{}.png'.format(suffix)), dpi=100)
        plt.clf()
        plt.close(fig)

        # fig2, ax3 = plt.subplots()
        # pr = [(recall[i], precision[i]) for i in range(len(thresh))]
        # ax3.plot(pr)
        # ax3.set_ylabel('threshold')
        # ax3.set_yticks(thresh)
        # fig2.set_size_inches(18.5, 10.5)
        # fig2.savefig(os.path.join(self.log_dir, 'other-pr-curve-{}.png'.format(suffix)), dpi=100)

        recall = np.array(recall)
        recall, uniq_idx = np.unique(recall, return_index=True)
        true_positive_counts = np.array(true_positive_counts)[uniq_idx]
        false_positive_counts = np.array(false_positive_counts)[uniq_idx]
        true_negative_counts = np.array(true_negative_counts)[uniq_idx]
        false_negative_counts = np.array(false_negative_counts)[uniq_idx]
        precision = np.array(precision)[uniq_idx]

        idxs = np.argsort(recall)[::-1]
        true_positive_counts = true_positive_counts[idxs].tolist()
        false_positive_counts = false_positive_counts[idxs].tolist()
        true_negative_counts = true_negative_counts[idxs].tolist()
        false_negative_counts = false_negative_counts[idxs].tolist()
        precision = precision[idxs].tolist()
        recall = recall[idxs].tolist()

        self.add_pr_curve_raw(true_positive_counts, false_positive_counts, true_negative_counts, false_negative_counts,
                              precision, recall, step, name)

    def add_pr_curve_raw(self, true_positive_counts, false_positive_counts, true_negative_counts, false_negative_counts,
                         precision, recall, step=None, name='ROC'):
        if not step:
            step = self.global_step

        num_thresholds = len(true_positive_counts)
        self.writer.add_pr_curve_raw(name, true_positive_counts, false_positive_counts, true_negative_counts,
                                     false_negative_counts, precision, recall, step, num_thresholds,
                                     weights=None)

    def add_pr_curve(self, tag, labels, predictions):
        self.writer.add_pr_curve(tag, labels, predictions, global_step=self.global_step)


class WandbLogger(TensorboardLogger):

    def __init__(self, log_dir, args={}):
        self.log_dir = self._prepare_log_dir(log_dir)

        self.args = args
        self.config= os.path.join(os.path.join(os.getcwd(), self.log_dir), 'config.yaml')
        save_yaml(self.config, self.args)

        config_dictionary = dict(
            yaml = self.config
        )
        logging.info("writing config file to: %s ", self.config)

        wandb.init(dir=self.log_dir, name=args.get("super_fancy_new_name", None), config=config_dictionary) 

        # self.log_dir = os.path.join(wandb.run.dir, "wandb") # wandb bug: inits a wandb directory in the log dir

        self.update_config({'checkpoint_dir': os.path.join(os.getcwd(), self.log_dir)})
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.global_step = 0
        self.epoch = None
    
    def update_config(self, argdict):
        wandb.config.update(argdict)

    def get_config(self, arg):
        return getattr(wandb.config, arg, None)
        
    def watch_model(self, model):
        wandb.watch(model)

    def log_dict_to_wandb(self, dic):
        wandb.log(dic)

    def log_value(self, name, value, step=None, commit=None):
        super().log_value(name, value, step=step)
        wandb.log({name : value}, commit=commit)

    def log_table(self, tab, key, columns):
        table = wandb.Table(data=tab, columns=columns)
        wandb.log({key :  table})

    def finish(self):
        wandb.finish()
    
    def save_file(self, filename):
        wandb.save(filename)
        
    def log_plots(self, xs, ys, keys, xname, title, custom_id):
        plt.figure()
        plt.plot(xs, np.transpose(np.array(ys)))
        plt.legend(keys)
        plt.xlabel(xname)
        plt.title(title)
        plt.grid()
        wandb.log({custom_id: wandb.Image(plt)})
        plt.clf()

class GPU:
    device = torch.device('cpu')

    @staticmethod
    def get_free_gpu(memory=1000):
        skinner_map = {0: 2, 1: 0, 2: 1, 3: 3}
        a = os.popen("/usr/bin/nvidia-smi | grep 'MiB /' | awk -e '{print $9}' | sed -e 's/MiB//'")

        free_memory = []
        while 1:
            line = a.readline()
            if not line:
                break
            free_memory.append(int(line))

        gpu = np.argmin(free_memory)
        if free_memory[gpu] < memory:
            if socket.gethostname() == "skinner":
                for k, v in skinner_map.items():
                    if v == gpu:
                        return k
            return gpu

        logging.error('No free GPU available.')
        exit(1)

    @classmethod
    def set(cls, gpuid, memory=1000):
        gpuid = int(gpuid)
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            logging.info("searching for free GPU")
            if gpuid == -1:
                gpuid = GPU.get_free_gpu(memory)
            os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpuid)
            if torch.cuda.device_count() == 1:  # sometimes this does not work
                torch.cuda.set_device(0)
            else:
                torch.cuda.set_device(int(gpuid))
        else:
            gpuid = os.environ['CUDA_VISIBLE_DEVICES']
            logging.info('taking GPU {} as specified in envorionment variable'.format(gpuid))
            torch.cuda.set_device(0)

        cls.device = torch.device('cuda:{}'.format(torch.cuda.current_device()))

        logging.info('Using GPU {}'.format(gpuid))
        return gpuid


class Timer:
    def __init__(self):
        self._start = time.time()

    def __str__(self):
        end = time.time()
        hours, rem = divmod(end - self._start, 3600)
        minutes, seconds = divmod(rem, 60)
        return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)

    def __call__(self):
        return self.__str__()


def load_config(args):
    args = args.__dict__
    c = args['config']
    config = load_yaml(c)

    # _, _l = {}, {}
    # with open(c[0], 'r') as f:
    #     exec(f.read(), globals(), _l)

    # config = _l.get('config__', None)
    # assert config, 'config__ must be set in config file'

    if type(config) != list:
        config = [config]
        
    argdicts = []

    for d in config:
        myargs = deepcopy(args)
        dict_merge(myargs, d)
        argdicts.append(myargs)

    return argdicts

    # else:

    #     if len(c) > 1:
    #         d = getattr(config, c[1])()
    #     else:
    #         d = config()
    #     dict_merge(args, d)
    #     return args


def dict_merge(dct, merge_dct, verify=False):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :param verify: checks if no entry is added to the dictionary
    :return: None
    """
    #     dct = copy.copy(dct)
    changes_values = {}
    changes_lists = {}

    for k, _ in merge_dct.items():
        if verify:
            assert k in dct, 'key "{}" is not part of the default dict'.format(k)
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], collections.Mapping)):
            changes_lists[k] = dict_merge(dct[k], merge_dct[k], verify=verify)
        else:
            if k in dct and dct[k] != merge_dct[k]:
                changes_values[k] = merge_dct[k]

            dct[k] = merge_dct[k]

    _sorted = []
    for k, _ in dct.items():
        if k in changes_values:
            _sorted.append((k, changes_values[k]))
        elif k in changes_lists:
            _sorted.extend(changes_lists[k])

    return _sorted


def json_hash(d):
    from hashlib import sha1
    assert d is not None, "Cannot hash None!"

    def hashnumpy(a):
        if type(a) == dict:
            for k, v in a.items():
                a[k] = hashnumpy(v)

        if type(a) == list:
            for i, v in enumerate(a):
                a[i] = hashnumpy(v)

        if type(a) == np.ndarray:
            return sha1(a).hexdigest()

        if hasattr(a, '__dict__'):
            return hashnumpy(a.__dict__)

        return a

    return sha1(json.dumps(hashnumpy(d), sort_keys=True).encode()).hexdigest()


class ColorString:

    @classmethod
    def color(cls, string, color):
        return color + string + '\033[0m'

    @classmethod
    def magenta(cls, string):
        return cls.color(string, '\033[95m')

    @classmethod
    def blue(cls, string):
        return cls.color(string, '\033[94m')

    @classmethod
    def green(cls, string):
        return cls.color(string, '\033[92m')

    @classmethod
    def yellow(cls, string):
        return cls.color(string, '\033[93m')

    @classmethod
    def fail(cls, string):
        return cls.color(string, '\033[91m')

    @classmethod
    def bold(cls, string):
        return cls.color(string, '\033[1m')

    @classmethod
    def underline(cls, string):
        return cls.color(string, '\033[4m')
