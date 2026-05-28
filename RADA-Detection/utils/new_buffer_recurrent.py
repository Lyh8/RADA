from collections import namedtuple
import numpy as np
import torch

# 扩展命名元组，包含last_onehot_a_n字段
RecurrentBatch = namedtuple('RecurrentBatch', 'o a r d m last_onehot_a_n')


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def as_probas(positive_values: np.array) -> np.array:
    return positive_values / np.sum(positive_values)


def as_tensor_on_device(np_array: np.array):
    return torch.tensor(np_array).float().to(get_device())


class MyRecurrentReplayBuffer:
    """Use this version when num_bptt == max_episode_len"""

    def __init__(
            self,
            o_dim,
            a_dim,
            max_episode_len,  # 同时作为num_bptt
            capacity,
            batch_size,
            n_agents,  # 智能体数量（动态参数）
            action_dim,  # 单个动作的维度（动态参数）
            segment_len=None
    ):
        # 初始化原有数据结构
        self.o = np.zeros((capacity, max_episode_len + 1, o_dim))
        self.a = np.zeros((capacity, max_episode_len, a_dim))
        self.r = np.zeros((capacity, max_episode_len, 1))
        self.d = np.zeros((capacity, max_episode_len, 1))
        self.m = np.zeros((capacity, max_episode_len, 1))

        # 新增：初始化last_onehot_a_n缓冲区（动态形状）
        # 形状：(capacity, max_episode_len + 1, n_agents, action_dim)
        self.last_onehot_a_n = np.zeros((capacity, max_episode_len + 1, n_agents, action_dim))

        self.ep_len = np.zeros((capacity,))
        self.ready_for_sampling = np.zeros((capacity,))

        # 指针
        self.episode_ptr = 0
        self.time_ptr = 0

        # 跟踪器
        self.starting_new_episode = True
        self.num_episodes = 0

        # 超参数（保存动态形状参数）
        self.capacity = capacity
        self.o_dim = o_dim
        self.a_dim = a_dim
        self.batch_size = batch_size
        self.max_episode_len = max_episode_len
        self.n_agents = n_agents  # 智能体数量
        self.action_dim = action_dim  # 单个动作维度
        self.last_onehot_a_n_shape = (n_agents, action_dim)  # 独热编码形状

        if segment_len is not None:
            assert max_episode_len % segment_len == 0

        self.segment_len = segment_len

    def push(self, o, a, r, no, d, cutoff, last_onehot_a_n):
        """
        新增参数：
        last_onehot_a_n: 形状为(n_agents, action_dim)的独热编码动作
        """
        if self.starting_new_episode:
            # 新episode开始时清零当前槽位
            self.o[self.episode_ptr] = 0
            self.a[self.episode_ptr] = 0
            self.r[self.episode_ptr] = 0
            self.d[self.episode_ptr] = 0
            self.m[self.episode_ptr] = 0
            self.last_onehot_a_n[self.episode_ptr] = 0  # 清零独热编码缓冲区
            self.ep_len[self.episode_ptr] = 0
            self.ready_for_sampling[self.episode_ptr] = 0

            self.starting_new_episode = False

        # 填充当前时间步的数据
        self.o[self.episode_ptr, self.time_ptr] = o
        self.a[self.episode_ptr, self.time_ptr] = a.cpu().numpy() if isinstance(a, torch.Tensor) else a
        self.r[self.episode_ptr, self.time_ptr] = r
        self.d[self.episode_ptr, self.time_ptr] = d
        self.m[self.episode_ptr, self.time_ptr] = 1

        # last_onehot_a_n 是 t-1 时刻的动作，对应 obs t
        # [修复] 检查 last_onehot_a_n 是否为 Tensor，如果是才转换，否则直接赋值
        self.last_onehot_a_n[self.episode_ptr, self.time_ptr] = last_onehot_a_n.cpu().numpy() if isinstance(last_onehot_a_n, torch.Tensor) else last_onehot_a_n

        self.ep_len[self.episode_ptr] += 1

        if d or cutoff:
            # 填充终止状态的下一个观测
            self.o[self.episode_ptr, self.time_ptr + 1] = no
            # 填充最后一个动作
            # self.last_onehot_a_n[self.episode_ptr, self.time_ptr + 1] = ... (取决于是否有下一个动作)
            self.ready_for_sampling[self.episode_ptr] = 1

            # 重置指针
            self.episode_ptr = (self.episode_ptr + 1) % self.capacity
            self.time_ptr = 0

            # 更新跟踪器
            self.starting_new_episode = True
            if self.num_episodes < self.capacity:
                self.num_episodes += 1
        else:
            # 推进时间指针
            self.time_ptr += 1

    def sample(self):
        assert self.num_episodes >= self.batch_size

        # 采样episode索引
        options = np.where(self.ready_for_sampling == 1)[0]
        ep_lens_of_options = self.ep_len[options]
        probas_of_options = as_probas(ep_lens_of_options)
        choices = np.random.choice(options, p=probas_of_options, size=self.batch_size)

        ep_lens_of_choices = self.ep_len[choices]

        if self.segment_len is None:
            # 全序列采样（不截断）
            max_ep_len_in_batch = int(np.max(ep_lens_of_choices))

            # 提取对应的数据片段
            o = self.o[choices][:, :max_ep_len_in_batch + 1, :]
            a = self.a[choices][:, :max_ep_len_in_batch, :]
            r = self.r[choices][:, :max_ep_len_in_batch, :]
            d = self.d[choices][:, :max_ep_len_in_batch, :]
            m = self.m[choices][:, :max_ep_len_in_batch, :]
            # 提取独热编码动作，形状为(batch_size, max_ep_len_in_batch, n_agents, action_dim)
            last_onehot_a_n = self.last_onehot_a_n[choices][:, :max_ep_len_in_batch + 1, :, :]

            # 转换为设备上的张量
            o = as_tensor_on_device(o).view(self.batch_size, max_ep_len_in_batch + 1, self.o_dim)
            a = as_tensor_on_device(a).view(self.batch_size, max_ep_len_in_batch, self.a_dim)
            r = as_tensor_on_device(r).view(self.batch_size, max_ep_len_in_batch, 1)
            d = as_tensor_on_device(d).view(self.batch_size, max_ep_len_in_batch, 1)
            m = as_tensor_on_device(m).view(self.batch_size, max_ep_len_in_batch, 1)
            # 独热编码张量形状保持为(batch_size, max_ep_len_in_batch, n_agents, action_dim)
            last_onehot_a_n = as_tensor_on_device(last_onehot_a_n).view(
                self.batch_size, max_ep_len_in_batch + 1, self.n_agents, self.action_dim
            )

            return RecurrentBatch(o, a, r, d, m, last_onehot_a_n)

        else:
            # 分段采样（截断BPTT）
            num_segments_for_each_item = np.ceil(ep_lens_of_choices / self.segment_len).astype(int)

            o = self.o[choices]
            a = self.a[choices]
            r = self.r[choices]
            d = self.d[choices]
            m = self.m[choices]
            last_onehot_a_n = self.last_onehot_a_n[choices]  # 获取独热编码数据

            # 初始化分段缓冲区
            o_seg = np.zeros((self.batch_size, self.segment_len + 1, self.o_dim))
            a_seg = np.zeros((self.batch_size, self.segment_len, self.a_dim))
            r_seg = np.zeros((self.batch_size, self.segment_len, 1))
            d_seg = np.zeros((self.batch_size, self.segment_len, 1))
            m_seg = np.zeros((self.batch_size, self.segment_len, 1))
            # 独热编码分段缓冲区，形状为(batch_size, segment_len, n_agents, action_dim)
            last_onehot_a_n_seg = np.zeros((self.batch_size, self.segment_len + 1, self.n_agents, self.action_dim))

            for i in range(self.batch_size):
                start_idx = np.random.randint(num_segments_for_each_item[i]) * self.segment_len
                o_seg[i] = o[i][start_idx:start_idx + self.segment_len + 1]
                a_seg[i] = a[i][start_idx:start_idx + self.segment_len]
                r_seg[i] = r[i][start_idx:start_idx + self.segment_len]
                d_seg[i] = d[i][start_idx:start_idx + self.segment_len]
                m_seg[i] = m[i][start_idx:start_idx + self.segment_len]
                # 提取独热编码分段
                last_onehot_a_n_seg[i] = last_onehot_a_n[i][start_idx:start_idx + self.segment_len + 1]

            # 转换为设备上的张量
            o_seg = as_tensor_on_device(o_seg)
            a_seg = as_tensor_on_device(a_seg)
            r_seg = as_tensor_on_device(r_seg)
            d_seg = as_tensor_on_device(d_seg)
            m_seg = as_tensor_on_device(m_seg)
            # 独热编码分段张量
            last_onehot_a_n_seg = as_tensor_on_device(last_onehot_a_n_seg)

            return RecurrentBatch(o_seg, a_seg, r_seg, d_seg, m_seg, last_onehot_a_n_seg)

    def reset(self):
        # 重置所有缓冲区（包括独热编码数据）
        self.o = np.zeros((self.capacity, self.max_episode_len + 1, self.o_dim))
        self.a = np.zeros((self.capacity, self.max_episode_len, self.a_dim))
        self.r = np.zeros((self.capacity, self.max_episode_len, 1))
        self.d = np.zeros((self.capacity, self.max_episode_len, 1))
        self.m = np.zeros((self.capacity, self.max_episode_len, 1))
        self.last_onehot_a_n = np.zeros((self.capacity, self.max_episode_len + 1, self.n_agents, self.action_dim))
        self.ep_len = np.zeros((self.capacity,))
        self.ready_for_sampling = np.zeros((self.capacity,))

        # 重置指针和跟踪器
        self.episode_ptr = 0
        self.time_ptr = 0
        self.starting_new_episode = True
        self.num_episodes = 0

