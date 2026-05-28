import torch as th
class RunningMeanStd:
    def __init__(self, shape=(), device="cpu"):
        self.mean = th.zeros(shape, device=device)
        self.var = th.ones(shape, device=device)
        self.count = 1e-4
    def update(self, x):
        batch_mean = th.mean(x, dim=0)
        batch_var = th.var(x, dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)
    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count