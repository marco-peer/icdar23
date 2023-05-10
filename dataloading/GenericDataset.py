from glob import glob
class ImageDataset:

    def __init__(self, root, loader, transform):
        self.root = root
        self.loader = loader
        self.transform = transform

        self.imgs = list(glob(f'{self.root}/*'))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.loader(self.imgs[idx])
        img = img.convert('RGB')
        return self.transform(img)

class FilepathImageDataset:

    def __init__(self, filepaths, loader, transform):
        self.loader = loader
        self.transform = transform

        self.imgs = filepaths

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.loader(self.imgs[idx])
        img = img.convert('RGB')
        return self.transform(img)
