import torch as th
import numpy as np
from types import SimpleNamespace as SN

class EpisodeBatch:
    def __init__(self, scheme, groups, batch_size, max_seq_length, data=None, preprocess=None, device="cpu"):
        self.scheme = scheme.copy()
        self.groups = groups
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.preprocess = preprocess
        self.device = device

        if data is not None:
            self.data = data
        else:
            self.data = SN()
            self.data.transition_data = {}
            self.data.episode_data = {}
            self._setup_data(self.scheme, self.groups, batch_size, max_seq_length, self.preprocess)

    def _setup_data(self, scheme, groups, batch_size, max_seq_length, preprocess):
        if preprocess is not None:
            for k in preprocess:
                k_out, transforms = preprocess[k]
                vshape = scheme[k]["vshape"]
                dtype = scheme[k].get("dtype", th.float32)
                for transform in transforms:
                    vshape, dtype = transform.infer_output_info(vshape, dtype)
                scheme[k_out] = {"vshape": vshape, "dtype": dtype}
                if "group" in scheme[k]: scheme[k_out]["group"] = scheme[k]["group"]
                if "episode_const" in scheme[k]: scheme[k_out]["episode_const"] = scheme[k]["episode_const"]

        if "filled" not in scheme:
            self.data.transition_data["filled"] = th.zeros((batch_size, max_seq_length, 1), dtype=th.long, device=self.device)

        for field_key, field_info in scheme.items():
            assert "vshape" in field_info
            vshape = field_info["vshape"]
            dtype = field_info.get("dtype", th.float32)
            if isinstance(vshape, int): vshape = (vshape,)
            if groups:
                if "group" in field_info:
                    vshape = (groups[field_info["group"]], *vshape)
            self.data.transition_data[field_key] = th.zeros((batch_size, max_seq_length, *vshape), dtype=dtype, device=self.device)

    def update(self, data, bs=slice(None), ts=slice(None), mark_filled=True):
        slices = self._parse_slices((bs, ts))
        if mark_filled:
            if "filled" in self.data.transition_data:
                self.data.transition_data["filled"][slices] = 1

        for k, v in data.items():
            if k in self.data.transition_data:
                target = self.data.transition_data[k]
                v_tensor = th.as_tensor(v, dtype=target.dtype, device=self.device)
                v_tensor_reshaped = v_tensor.view_as(target[slices])
                target[slices] = v_tensor_reshaped

                if self.preprocess and k in self.preprocess:
                    new_k = self.preprocess[k][0]
                    transforms = self.preprocess[k][1]
                    v_processed = v_tensor_reshaped
                    for transform in transforms:
                        v_processed = transform.transform(v_processed)
                    target_proc = self.data.transition_data[new_k]
                    target_proc[slices] = v_processed.view_as(target_proc[slices])

    def _parse_slices(self, items): return items

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.data.transition_data: return self.data.transition_data[item]
            elif item in self.data.episode_data: return self.data.episode_data[item]
            else: raise ValueError(f"Key {item} not found in EpisodeBatch")
        else:
            new_data = SN()
            new_data.transition_data = {}
            new_data.episode_data = {}
            new_batch = EpisodeBatch(self.scheme, self.groups, self.batch_size, self.max_seq_length, data=new_data, preprocess=self.preprocess, device=self.device)
            if isinstance(item, tuple): batch_slice = item[0]
            else: batch_slice = item
            for k, v in self.data.transition_data.items(): new_batch.data.transition_data[k] = v[item]
            for k, v in self.data.episode_data.items(): new_batch.data.episode_data[k] = v[batch_slice]
            if new_batch.data.transition_data:
                ref_k = list(new_batch.data.transition_data.keys())[0]
                new_batch.batch_size = new_batch.data.transition_data[ref_k].shape[0]
                new_batch.max_seq_length = new_batch.data.transition_data[ref_k].shape[1]
            return new_batch

    def max_t_filled(self): return self.max_seq_length
    def to(self, device):
        for k, v in self.data.transition_data.items(): self.data.transition_data[k] = v.to(device)
        self.device = device

class ReplayBuffer(EpisodeBatch):
    def __init__(self, scheme, groups, buffer_size, max_seq_length, preprocess=None, device="cpu"):
        super(ReplayBuffer, self).__init__(scheme, groups, buffer_size, max_seq_length, preprocess=preprocess, device=device)
        self.buffer_size = buffer_size
        self.buffer_index = 0
        self.episodes_in_buffer = 0

    def insert_episode_batch(self, ep_batch):
        if self.buffer_index + ep_batch.batch_size <= self.buffer_size:
            self.update(ep_batch.data.transition_data, bs=slice(self.buffer_index, self.buffer_index + ep_batch.batch_size))
            self.buffer_index = (self.buffer_index + ep_batch.batch_size)
            self.episodes_in_buffer = max(self.episodes_in_buffer, self.buffer_index)
            if self.buffer_index == self.buffer_size: self.buffer_index = 0
        else: pass

    def can_sample(self, batch_size): return self.episodes_in_buffer >= batch_size
    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        indices = np.random.choice(self.episodes_in_buffer, batch_size, replace=False)
        new_batch = EpisodeBatch(self.scheme, self.groups, batch_size, self.max_seq_length, preprocess=None, device=self.device)
        for k in self.data.transition_data: new_batch.data.transition_data[k] = self.data.transition_data[k][indices]
        return new_batch