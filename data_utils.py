import os, json, urllib.request
from PIL import Image
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

class CustomImageNetDataV1(Dataset): # Inheriting from Dataset  a must to work seamlessly with PyTorch’s data loading ecosystem (DataLoader, Subset, ConcatDataset). PyTorch expects a dataset object and by passing Dataset can verify the output is a dataset object
    def __init__(self, img_dir: str, img_type: str):
        self.img_dir = img_dir
        self.img_type = img_type  # Triggers the setter

        self.class_to_idx = self._load_imagenet_class_mapping()
        self.image_paths, self.labels = self._load_image_paths_and_labels()

    # managed attributes
    @property  # turns a method into a read-only attribute and allow using setter or change the implementation without breaking external code for valiation
    def img_type(self):
        return self._img_type

    @img_type.setter # allows to change the attribute after intialization as well as validating img_type as it must be
    def img_type(self, value):
        if not isinstance(value, str):
            raise TypeError("img_type must be a string")
        value = value.lower()
        allowed_types = {"original", "reconstructed"}
        if value not in allowed_types:
            raise ValueError(f"img_type must be one of {allowed_types}, but got '{value}'")
        self._img_type = value

    @property
    def transform(self):
        if self.img_type == "original":
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        elif self.img_type == "reconstructed":
            return transforms.Compose([
                transforms.Resize((80, 80)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
            ])

    def _load_imagenet_class_mapping(self): # leading underscore signals its a non-public method and intended to be used internally
        class_index_url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
        with urllib.request.urlopen(class_index_url) as url:
            class_idx = json.loads(url.read().decode())
        # Maps WordNet ID (e.g. 'n01440764') to integer index
        return {value[0]: int(key) for key, value in class_idx.items()}

    def _load_image_paths_and_labels(self):
        image_paths = []
        labels = []
        for label_name in sorted(os.listdir(self.img_dir)):
            label_path = os.path.join(self.img_dir, label_name)
            if not os.path.isdir(label_path):
                continue
            label_idx = self.class_to_idx.get(label_name)
            if label_idx is None:
                print(f"Warning: Label '{label_name}' not found in ImageNet class index.")
                continue
            for fname in os.listdir(label_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(label_path, fname))
                    labels.append(label_idx)
        return image_paths, labels

    # Dunder Methods (special methods that Python uses to implement built-in behavior). They are implemented automatically and behinde the sense
    ## Allows python to identify a length for the created object -- you can use len(obj) and it returns the length by calling obj.__len__() method; otherwise, if your defined, custom, object has a length but you didn't define the __len__ method it returns an error.
    def __len__(self):
        return len(self.image_paths)

    ## Allows python to identify indexing for the created object -- you can use obj[n] and as n < len(obj) it returns corresponding element by calling obj.__getitem__(n)
    def __getitem__(self, idx):
        '''
        Return a tuple as calling obj[n]

        Argument:
        idx: idx is exactly the same as passed index n
        '''
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(image_path).convert("RGB") # ?convert?
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
        image = self.transform(image)
        return image, label


class CustomImageNetDataV2(ImageFolder):
    '''
    General-purpose ImageNet-style dataset loader.
    Supports folders labeled by WordNet ID, class name, or integer ID.
    Remaps labels to official ImageNet indices when possible.
    '''

    def __init__(self, image_dir: str, image_type: str, folder_label: str = 'word_net_id'):
        self._img_type = image_type
        self._folder_label = folder_label

        transform = self._get_transform()
        super().__init__(root=image_dir, transform=transform)

        # Load class mapping
        self.idx_map = self._build_index_mapping()

        # Remap labels
        self.samples = [(path, self._map_label(self.classes[label])) for path, label in self.samples]

    def _get_transform(self):
        if self._img_type == "original":
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        elif self._img_type == "reconstructed":
            return transforms.Compose([
                transforms.Resize((80, 80)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
            ])
        else:
            raise ValueError(f"Unsupported image_type '{self._img_type}'")

    def _build_index_mapping(self):
        url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
        with urllib.request.urlopen(url) as response:
            class_idx = json.loads(response.read().decode())

        if self._folder_label == 'word_net_id':
            return {v[0]: int(k) for k, v in class_idx.items()}
        elif self._folder_label == 'class_id':
            return {v[1]: int(k) for k, v in class_idx.items()}
        elif self._folder_label == 'int_id':
            return {cls: int(cls) for cls in self.classes if cls.isdigit()}
        else:
            raise ValueError(f"Unsupported folder_label '{self._folder_label}'")

    def _map_label(self, folder_name):
        try:
            return self.idx_map[folder_name]
        except KeyError:
            raise ValueError(f"Folder name '{folder_name}' not found in mapping for label type '{self._folder_label}'")

    @property
    def img_type(self):
        return self._img_type

    @property
    def folder_label(self):
        return self._folder_label        


class CustomLoader:
    def __init__(self, dataset, batch_size, threads = 4, shuffle = True, distributed = False, world_size = None, rank = None,  split = False,
                 val=False, train_size=0.7, test_size=0.5, train_size_adj=0.1):
        self._dataset = dataset
        self._batch_size = batch_size
        self._threads = threads
        self._shuffle = shuffle
        self._labels = [label for _, label in dataset.samples]

        # Check if distributed training is active
        self._distributed = distributed #dist.is_available() and dist.is_initialized()
        self._world_size = world_size
        self._rank = rank

        if split:
            self._train_loader, self._val_loader, self._test_loader = self._create_data_loaders(
                val, train_size, test_size, train_size_adj)
            self._data_loader = None
        else:
            self._data_loader = self._create_full_loader()
            self._train_loader = self._val_loader = self._test_loader = None

    @property
    def labels(self):
        return self._labels

    @property
    def train_loader(self):
        return self._train_loader

    @property
    def val_loader(self):
        return self._val_loader

    @property
    def test_loader(self):
        return self._test_loader

    @property
    def data_loader(self):
        return self._data_loader

    def _create_full_loader(self):
        sampler = DistributedSampler(self._dataset, num_replicas=self._world_size, rank=self._rank) if self._distributed else None
        return DataLoader(
            dataset=self._dataset,
            batch_size=self._batch_size,
            shuffle=(sampler is None and self._shuffle),
            sampler=sampler,
            num_workers=self._threads,
            pin_memory=True
        )

    def _create_data_loaders(self, val, train_size, test_size, train_size_adj):
        if not val:
            splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_size + train_size_adj, random_state=42)
            train_idx, test_idx = next(splitter.split(self._dataset.samples, self._labels))

            train_dataset = Subset(self._dataset, train_idx)
            test_dataset = Subset(self._dataset, test_idx)

            train_sampler = DistributedSampler(train_dataset, num_replicas=self._world_size, rank=self._rank) if self._distributed else None
            test_sampler = DistributedSampler(test_dataset, num_replicas=self._world_size, rank=self._rank) if self._distributed else None

            return (
                DataLoader(train_dataset, batch_size=self._batch_size, shuffle=(train_sampler is None and self._shuffle),
                           sampler=train_sampler, num_workers=self._threads, pin_memory=True),
                None,
                DataLoader(test_dataset, batch_size=self._batch_size, shuffle=(test_sampler is None and self._shuffle),
                           sampler=test_sampler, num_workers=self._threads, pin_memory=True)
            )
        else:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
            train_idx, temp_idx = next(splitter.split(self._dataset.samples, self._labels))

            temp_labels = [self._labels[i] for i in temp_idx]
            splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
            val_idx, test_idx = next(splitter2.split([self._dataset.samples[i] for i in temp_idx], temp_labels))

            train_dataset = Subset(self._dataset, train_idx)
            val_dataset = Subset(self._dataset, [temp_idx[i] for i in val_idx])
            test_dataset = Subset(self._dataset, [temp_idx[i] for i in test_idx])

            train_sampler = DistributedSampler(train_dataset, num_replicas=self._world_size, rank=self._rank) if self._distributed else None
            val_sampler = DistributedSampler(val_dataset, num_replicas=self._world_size, rank=self._rank) if self._distributed else None
            test_sampler = DistributedSampler(test_dataset, num_replicas=self._world_size, rank=self._rank) if self._distributed else None

            return (
                DataLoader(train_dataset, batch_size=self._batch_size, shuffle=(train_sampler is None and self._shuffle),
                           sampler=train_sampler, num_workers=self._threads, pin_memory=True),
                DataLoader(val_dataset, batch_size=self._batch_size, shuffle=(val_sampler is None and self._shuffle),
                           sampler=val_sampler, num_workers=self._threads, pin_memory=True),
                DataLoader(test_dataset, batch_size=self._batch_size, shuffle=(test_sampler is None and self._shuffle),
                           sampler=test_sampler, num_workers=self._threads, pin_memory=True)
            )

# train_loader = DataLoader(
#     dataset=train_dataset,
#     batch_size=32,
#     shuffle=True,
#     num_workers=4,
#     pin_memory=True  # Recommended for CUDA
# )

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # Move Data to CUDA in Training Loop
# for images, labels in train_loader:
#     images = images.to(device)
#     labels = labels.to(device)
#     # Forward pass, loss, etc.
