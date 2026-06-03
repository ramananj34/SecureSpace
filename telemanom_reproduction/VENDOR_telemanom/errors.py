from __future__ import annotations
import numpy as np
import pandas as pd
import more_itertools as mit

class Errors:
    def __init__(self, channel, config, verbose: bool = False):
        self.config = config
        self.window_size = self.config.window_size
        self.n_windows = int((channel.y_test.shape[0] - (self.config.batch_size * self.window_size)) / self.config.batch_size)
        self.i_anom = np.array([])
        self.E_seq = []
        self.anom_scores = []
        self.e = [abs(y_h - y_t[0]) for y_h, y_t in zip(channel.y_hat, channel.y_test)]
        smoothing_window = int(self.config.batch_size * self.config.window_size * self.config.smoothing_perc)
        if not len(channel.y_hat) == len(channel.y_test):
            raise ValueError(f'len(y_hat) != len(y_test): {len(channel.y_hat)}, {len(channel.y_test)}')
        self.e_s = pd.DataFrame(self.e).ewm(span=smoothing_window).mean().values.flatten()
        if not channel.id == 'C-2':
            self.e_s[:self.config.l_s] = ([np.mean(self.e_s[:self.config.l_s * 2])] * self.config.l_s)
        self.normalized = float(np.mean(self.e / np.ptp(channel.y_test)))
        if verbose:
            print(f"[Errors] normalized prediction error: {self.normalized:.2f}")

    def adjust_window_size(self, channel):
        while self.n_windows < 0:
            self.window_size -= 1
            self.n_windows = int((channel.y_test.shape[0] - (self.config.batch_size * self.window_size)) / self.config.batch_size)
            if self.window_size == 1 and self.n_windows < 0:
                raise ValueError(f'batch_size ({self.config.batch_size}) larger than 'f'y_test (len={channel.y_test.shape[0]}). Adjust config.')

    def merge_scores(self):
        pass

    def process_batches(self, channel):
        self.adjust_window_size(channel)
        for i in range(0, self.n_windows + 1):
            prior_idx = i * self.config.batch_size
            idx = (self.config.window_size * self.config.batch_size) + (i * self.config.batch_size)
            if i == self.n_windows:
                idx = channel.y_test.shape[0]
            window = ErrorWindow(channel, self.config, prior_idx, idx, self, i)
            window.find_epsilon()
            window.find_epsilon(inverse=True)
            window.compare_to_epsilon(self)
            window.compare_to_epsilon(self, inverse=True)
            if len(window.i_anom) == 0 and len(window.i_anom_inv) == 0:
                continue
            window.prune_anoms()
            window.prune_anoms(inverse=True)
            if len(window.i_anom) == 0 and len(window.i_anom_inv) == 0:
                continue
            window.i_anom = np.sort(np.unique(np.append(window.i_anom, window.i_anom_inv))).astype('int')
            window.score_anomalies(prior_idx)
            self.i_anom = np.append(self.i_anom, window.i_anom + prior_idx)
            self.anom_scores = self.anom_scores + window.anom_scores
        if len(self.i_anom) > 0:
            groups = [list(group) for group in mit.consecutive_groups(self.i_anom)]
            self.E_seq = [(int(g[0]), int(g[-1])) for g in groups if not g[0] == g[-1]]
            self.E_seq = [(e_seq[0] + self.config.l_s, e_seq[1] + self.config.l_s) for e_seq in self.E_seq]
            self.merge_scores()

class ErrorWindow:

    def __init__(self, channel, config, start_idx, end_idx, errors, window_num):
        self.i_anom = np.array([])
        self.E_seq = np.array([])
        self.non_anom_max = -1000000
        self.i_anom_inv = np.array([])
        self.E_seq_inv = np.array([])
        self.non_anom_max_inv = -1000000
        self.config = config
        self.anom_scores = []
        self.window_num = window_num
        self.sd_lim = 12.0
        self.sd_threshold = self.sd_lim
        self.sd_threshold_inv = self.sd_lim
        self.e_s = errors.e_s[start_idx:end_idx]
        self.mean_e_s = np.mean(self.e_s)
        self.sd_e_s = np.std(self.e_s)
        self.e_s_inv = np.array([self.mean_e_s + (self.mean_e_s - e) for e in self.e_s])
        self.epsilon = self.mean_e_s + self.sd_lim * self.sd_e_s
        self.epsilon_inv = self.mean_e_s + self.sd_lim * self.sd_e_s
        self.y_test = channel.y_test[start_idx:end_idx]
        self.sd_values = np.std(self.y_test)
        self.perc_high, self.perc_low = np.percentile(self.y_test, [95, 5])
        self.inter_range = self.perc_high - self.perc_low
        self.num_to_ignore = self.config.l_s * 2
        if len(channel.y_test) < 2500:
            self.num_to_ignore = self.config.l_s
        if len(channel.y_test) < 1800:
            self.num_to_ignore = 0

    def find_epsilon(self, inverse=False):
        e_s = self.e_s if not inverse else self.e_s_inv
        max_score = -10000000
        for z in np.arange(2.5, self.sd_lim, 0.5):
            epsilon = self.mean_e_s + (self.sd_e_s * z)
            pruned_e_s = e_s[e_s < epsilon]
            i_anom = np.argwhere(e_s >= epsilon).reshape(-1,)
            buffer = np.arange(1, self.config.error_buffer)
            i_anom = np.sort(np.concatenate((i_anom,np.array([i + buffer for i in i_anom]).flatten(),np.array([i - buffer for i in i_anom]).flatten())))
            i_anom = i_anom[(i_anom < len(e_s)) & (i_anom >= 0)]
            i_anom = np.sort(np.unique(i_anom))
            if len(i_anom) > 0:
                groups = [list(group) for group in mit.consecutive_groups(i_anom)]
                E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]
                mean_perc_decrease = (self.mean_e_s - np.mean(pruned_e_s)) / self.mean_e_s
                sd_perc_decrease = (self.sd_e_s - np.std(pruned_e_s)) / self.sd_e_s
                score = (mean_perc_decrease + sd_perc_decrease) / (len(E_seq) ** 2 + len(i_anom))
                if score >= max_score and len(E_seq) <= 5 and len(i_anom) < (len(e_s) * 0.5):
                    max_score = score
                    if not inverse:
                        self.sd_threshold = z
                        self.epsilon = self.mean_e_s + z * self.sd_e_s
                    else:
                        self.sd_threshold_inv = z
                        self.epsilon_inv = self.mean_e_s + z * self.sd_e_s

    def compare_to_epsilon(self, errors_all, inverse=False):
        e_s = self.e_s if not inverse else self.e_s_inv
        epsilon = self.epsilon if not inverse else self.epsilon_inv
        if not (self.sd_e_s > (.05 * self.sd_values) or max(self.e_s) > (.05 * self.inter_range)) or not max(self.e_s) > 0.05:
            return
        i_anom = np.argwhere((e_s >= epsilon) & (e_s > 0.05 * self.inter_range)).reshape(-1,)
        if len(i_anom) == 0:
            return
        buffer = np.arange(1, self.config.error_buffer + 1)
        i_anom = np.sort(np.concatenate((i_anom,np.array([i + buffer for i in i_anom]).flatten(),np.array([i - buffer for i in i_anom]).flatten())))
        i_anom = i_anom[(i_anom < len(e_s)) & (i_anom >= 0)]
        if self.window_num == 0:
            i_anom = i_anom[i_anom >= self.num_to_ignore]
        else:
            i_anom = i_anom[i_anom >= len(e_s) - self.config.batch_size]
        i_anom = np.sort(np.unique(i_anom))
        batch_position = self.window_num * self.config.batch_size
        window_indices = np.arange(0, len(e_s)) + batch_position
        adj_i_anom = i_anom + batch_position
        window_indices = np.setdiff1d(window_indices, np.append(errors_all.i_anom, adj_i_anom))
        candidate_indices = np.unique(window_indices - batch_position)
        non_anom_max = np.max(np.take(e_s, candidate_indices))
        groups = [list(group) for group in mit.consecutive_groups(i_anom)]
        E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]
        if inverse:
            self.i_anom_inv = i_anom
            self.E_seq_inv = E_seq
            self.non_anom_max_inv = non_anom_max
        else:
            self.i_anom = i_anom
            self.E_seq = E_seq
            self.non_anom_max = non_anom_max

    def prune_anoms(self, inverse=False):
        E_seq = self.E_seq if not inverse else self.E_seq_inv
        e_s = self.e_s if not inverse else self.e_s_inv
        non_anom_max = self.non_anom_max if not inverse else self.non_anom_max_inv
        if len(E_seq) == 0:
            return
        E_seq_max = np.array([max(e_s[e[0]:e[1] + 1]) for e in E_seq])
        E_seq_max_sorted = np.sort(E_seq_max)[::-1]
        E_seq_max_sorted = np.append(E_seq_max_sorted, [non_anom_max])
        i_to_remove = np.array([])
        for i in range(0, len(E_seq_max_sorted) - 1):
            if (E_seq_max_sorted[i] - E_seq_max_sorted[i + 1]) / E_seq_max_sorted[i] < self.config.p:
                i_to_remove = np.append(i_to_remove,np.argwhere(E_seq_max == E_seq_max_sorted[i]))
            else:
                i_to_remove = np.array([])
        i_to_remove[::-1].sort()
        if len(i_to_remove) > 0:
            E_seq = np.delete(E_seq, i_to_remove, axis=0)
        if len(E_seq) == 0 and inverse:
            self.i_anom_inv = np.array([])
            return
        elif len(E_seq) == 0 and not inverse:
            self.i_anom = np.array([])
            return
        indices_to_keep = np.concatenate([range(e_seq[0], e_seq[-1] + 1) for e_seq in E_seq])
        if not inverse:
            mask = np.isin(self.i_anom, indices_to_keep)
            self.i_anom = self.i_anom[mask]
        else:
            mask_inv = np.isin(self.i_anom_inv, indices_to_keep)
            self.i_anom_inv = self.i_anom_inv[mask_inv]

    def score_anomalies(self, prior_idx):
        groups = [list(group) for group in mit.consecutive_groups(self.i_anom)]
        for e_seq in groups:
            score_dict = {"start_idx": e_seq[0] + prior_idx,"end_idx": e_seq[-1] + prior_idx,"score": 0}
            score = max([abs(self.e_s[i] - self.epsilon) / (self.mean_e_s + self.sd_e_s)for i in range(e_seq[0], e_seq[-1] + 1)])
            inv_score = max([abs(self.e_s_inv[i] - self.epsilon_inv) / (self.mean_e_s + self.sd_e_s)for i in range(e_seq[0], e_seq[-1] + 1)])
            score_dict['score'] = max([score, inv_score])
            self.anom_scores.append(score_dict)