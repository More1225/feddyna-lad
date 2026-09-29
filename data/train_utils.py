import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any, Optional
import os


class LogDataset(Dataset):
    """
    HDFS log-anomaly dataset in the unified NPZ format.

    Dynamic padding is supported:
    - If the NPZ includes train_lengths/val_lengths/test_lengths, return the actual sequence lengths.
    - These lengths are used by pack_padded_sequence to handle variable-length sequences.
    """

    MODES = ['train', 'val', 'test']

    def __init__(self, npz_path: str, mode: str = 'train') -> None:
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Dataset file not found: {npz_path}")
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES} ; received {mode}")

        self.npz_path = npz_path
        self.mode = mode

        data = np.load(npz_path, allow_pickle=True)

        # Unified NPZ format
        if mode == 'train':
            self.features = data['train_features'].astype(np.float32)
            self.labels = data['train_labels'].astype(np.int64)
            self.lengths = data['train_lengths'].astype(np.int64) if 'train_lengths' in data.files else None
        elif mode == 'val':
            # Prefer the validation split; fall back to the test split when validation data is unavailable for backward compatibility.
            if 'val_features' in data:
                self.features = data['val_features'].astype(np.float32)
                self.labels = data['val_labels'].astype(np.int64)
                self.lengths = data['val_lengths'].astype(np.int64) if 'val_lengths' in data.files else None
            else:
                print("Warning: validation data (val_features/val_labels) was not found in the NPZ file; using the test split for validation.")
                self.features = data['test_features'].astype(np.float32)
                self.labels = data['test_labels'].astype(np.int64)
                self.lengths = data['test_lengths'].astype(np.int64) if 'test_lengths' in data.files else None
        elif mode == 'test':
            if 'test_features' in data:
                self.features = data['test_features'].astype(np.float32)
                self.labels = data['test_labels'].astype(np.int64)
                self.lengths = data['test_lengths'].astype(np.int64) if 'test_lengths' in data.files else None
            else:
                raise KeyError("Test data (test_features/test_labels) was not found in the NPZ file.")

        # Check whether dynamic padding is available.
        self.has_dynamic_padding = self.lengths is not None

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Return features, labels, and the actual sequence length.

        Returns:
            feature: (seq_len, embed_dim) feature tensor
            label: label value
            length: actual sequence length when dynamic padding is supported; otherwise None
        """
        feature = torch.FloatTensor(self.features[item])
        label = torch.LongTensor([self.labels[item]]).squeeze()

        if self.has_dynamic_padding:
            length = torch.LongTensor([self.lengths[item]]).squeeze()
            return feature, label, length
        else:
            return feature, label, None


class DatasetSplit(Dataset):
    """Custom dataset wrapper that returns indexed subsets."""

    def __init__(self, dataset: Dataset, idxs: List[int]) -> None:
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self) -> int:
        return len(self.idxs)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        result = self.dataset[self.idxs[item]]
        # Check whether the result is None.
        if result is None:
            # Return a valid default value.
            # Assume the dataset exposes its feature dimension.
            if hasattr(self.dataset, 'features') and len(self.dataset.features) > 0:
                # Use the first valid sample as the default.
                for idx in self.idxs:
                    sample = self.dataset[idx]
                    if sample is not None:
                        return sample
            # Create a default sample if no valid sample is found.
            print(f"Warning: DatasetSplit got None at index {item}, creating default sample")
            # Create a random sample as the default.
            feature_dim = getattr(self.dataset, 'feature_dim', 300)  # default feature dimension
            seq_len = getattr(self.dataset, 'seq_len', 75)  # default sequence length
            return (
                torch.zeros((seq_len, feature_dim)),
                torch.tensor(0),
                torch.tensor(seq_len)
            )
        return result


def collate_fn_with_lengths(batch):
    """
    Collate function with dynamic-padding support.

    Sort batch samples by actual length for pack_padded_sequence.

    Args:
        batch: [(feature, label, length), ...] list

    Returns:
        features: (batch, seq_len, embed_dim) tensor
        labels: (batch,) label tensor
        lengths: (batch,) actual sequence-length tensor (when dynamic padding is supported), or None otherwise
    """
    # Check whether sequence-length information is available.
    has_length = batch[0][2] is not None

    if has_length:
        # Sort by descending length as required by pack_padded_sequence.
        batch = sorted(batch, key=lambda x: x[2], reverse=True)

        features = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        lengths = torch.stack([item[2] for item in batch])

        return features, labels, lengths
    else:
        # Without dynamic padding, stack samples directly.
        features = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        return features, labels, None


def get_data(args: Any) -> Tuple[Dataset, Dataset, Dataset, Dict, Dict, Dict, np.ndarray]:
    """
    Load federated-learning data (unified NPZ format)
    Supports separate training, validation, and test splits.
    """
    if not hasattr(args, 'data_path'):
        raise ValueError("Missing required --data_path argument.")

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Dataset file not found: {args.data_path}")

    data = np.load(args.data_path, allow_pickle=True)

    dataset_train = LogDataset(args.data_path, mode='train')
    dataset_val = LogDataset(args.data_path, mode='val')
    dataset_test = LogDataset(args.data_path, mode='test')

    dict_users_train = data['dict_users_train'].item()

    # Load the validation partition, falling back to the test partition when unavailable.
    if 'dict_users_val' in data:
        dict_users_val = data['dict_users_val'].item()
    else:
        print("Warning: validation partition (dict_users_val) was not found in the NPZ file; using the test partition for validation.")
        dict_users_val = data['dict_users_test'].item()

    # Load the test partition.
    dict_users_test = data['dict_users_test'].item()

    labels = dataset_train.labels

    client_ids = data['client_ids'] if 'client_ids' in data.files else list(dict_users_train.keys())
    args.num_users = len(client_ids)

    return dataset_train, dataset_val, dataset_test, dict_users_train, dict_users_val, dict_users_test, labels


def prepare_dataloaders(
    dataset_train: Dataset,
    dict_users_train: Dict,
    dataset_test: Dataset,
    dict_users_test: Dict,
    args: Any,
    use_dynamic_padding: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare training and test data loaders for the specified client.

    Args:
        dataset_train: training dataset
        dict_users_train: client partition for the training split
        dataset_test: test dataset
        dict_users_test: client partition for the test split
        args: argument configuration
        use_dynamic_padding: whether to use dynamic padding (collate_fn_with_lengths)

    Returns:
        ldr_train: training data loader
        ldr_test: test data loader
    """

    train_subset = DatasetSplit(dataset_train, dict_users_train)
    test_subset = DatasetSplit(dataset_test, dict_users_test)

    # Select the collate function.
    collate_fn = collate_fn_with_lengths if use_dynamic_padding else None

    ldr_train = DataLoader(
        train_subset,
        batch_size=getattr(args, 'local_bs', getattr(args, 'batch_size', 64)),
        shuffle=True,
        num_workers=args.num_workers if hasattr(args, 'num_workers') else 0,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=getattr(args, 'drop_last', False),
        collate_fn=collate_fn,
    )

    ldr_test = DataLoader(
        test_subset,
        batch_size=getattr(args, 'local_bs', getattr(args, 'batch_size', 64)),
        shuffle=False,
        num_workers=args.num_workers if hasattr(args, 'num_workers') else 0,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=getattr(args, 'drop_last', False),
        collate_fn=collate_fn,
    )

    return ldr_train, ldr_test