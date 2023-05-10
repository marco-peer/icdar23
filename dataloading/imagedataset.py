import os
from PIL import Image  # reading image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """Image dataset."""

    def __init__(self, base_path, set, image_dir, label_dir, label_class, image_transform, label_transform, limit=None):
        """
        Args:
            base_path   (string): Path to the dataset root.
            set         (string): Set to use [train, test, val]
            image_dir   (string): Directory name of the images directories.
            label_dir   (string): Directory name of the labels directories.
            label_class (string): Label images to use
            image_transform   (callable, optional): Optional transform to be applied on images.
            label_transform   (callable, optional): Optional transform to be applied on labels.
            limit   (int): Optional limit data set size.
        """

        self.base_path = base_path
        self.set = set
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.label_class = label_class
        self.image_transform = image_transform
        self.label_transform = label_transform

        entries = []

        images_path = '{base}/{type}/{set}/'.format(base=self.base_path, type=self.image_dir, set=set)
        labels_path = '{base}/{type}/{set}/'.format(base=self.base_path, type=self.label_dir, set=set)

        for _, dirs, _ in os.walk(images_path):
            for directory in dirs:
                for _, _, files in os.walk(os.path.join(images_path, directory)):
                    for filename in files:
                        image_path = os.path.join(images_path, directory, filename)
                        label_name = filename.replace(self.image_dir, self.label_class)
                        label_path = os.path.join(labels_path, directory, label_name)
                        entries.append({'image': image_path, 'label': label_path})
                        if limit is not None and len(entries) >= limit:
                            break
        assert len(entries) > 0, 'No data was found!'
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        image = Image.open(entry['image'])
        image_label = Image.open(entry['label'])

        # apply transforms
        if self.image_transform:
            image = self.image_transform(image)
        if self.label_transform:
            image_label = self.label_transform(image_label)

        return image, image_label
