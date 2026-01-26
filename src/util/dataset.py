# Modified by: (c) 2024 Gustav Norén
# Original author: (c) 2023 Anubhab Ghosh
# This code is licensed under MIT licence (see LICENSE file for details)

import numpy as np
import torch.utils.data


class SeriesDataset(torch.utils.data.Dataset):
    def __init__(self, z_xy_dict):
        self.data_dict = z_xy_dict
        self.trajectory_length = z_xy_dict["basic_data_parameters"].trajectory_length

    def __len__(self):
        return len(self.data_dict["data_x"])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            "inputs": np.expand_dims(self.data_dict["data_y"][idx], axis=0),
            "targets": np.expand_dims(self.data_dict["data_x"][idx], axis=0),
        }

        return sample


def obtain_tr_val_test_idx(dataset, supervised=0.1, test=0.1, val=0.17, labels=None):
    num_samples = len(dataset)
    num_supervised = int(supervised * num_samples)
    num_test = int(test * num_samples)
    num_val = int(val * num_samples)
    num_train = num_samples - num_supervised - num_test - num_val

    while True:
        indices = torch.randperm(num_samples).tolist()
        tr_indices = indices[:num_train]
        val_indices = indices[num_train : num_train + num_val]
        test_indices = indices[num_train + num_val : num_train + num_val + num_test]
        supervised_indices = indices[num_train + num_val + num_test :]
        if (
            sum([labels[supervised_index] for supervised_index in supervised_indices])
            == len(supervised_indices) / 2
        ):
            break

    return supervised_indices, tr_indices, val_indices, test_indices


def my_collate_fn(batch):
    inputs = [item["inputs"] for item in batch]
    targets = [item["targets"] for item in batch]
    inputs = [torch.tensor(item, dtype=torch.float32) for item in inputs]
    targets = [torch.tensor(item, dtype=torch.float32) for item in targets]
    targets = torch.from_numpy(np.vstack(targets))
    inputs = torch.from_numpy(np.vstack(inputs))
    return (inputs, targets)


def get_dataloaders(
    dataset,
    batch_size,
    tr_indices,
    val_indices,
    test_indices=None,
    val_batch_size=None,
    te_batch_size=None,
):
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(tr_indices),
        num_workers=0,
        collate_fn=my_collate_fn,
    )

    if val_batch_size is None:
        val_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(val_indices),
            num_workers=0,
            collate_fn=my_collate_fn,
        )
    else:
        val_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=val_batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(val_indices),
            num_workers=0,
            collate_fn=my_collate_fn,
        )

    if te_batch_size is None:
        test_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(test_indices),
            num_workers=0,
            collate_fn=my_collate_fn,
        )
    else:
        test_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=te_batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(test_indices),
            num_workers=0,
            collate_fn=my_collate_fn,
        )

    return train_loader, val_loader, test_loader
