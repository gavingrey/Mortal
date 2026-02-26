import torch
import numpy as np

class RewardCalculator:
    def __init__(self, grp=None, pts=None, uniform_init=False, shared_stats=None):
        self.device = torch.device('cpu')
        self.grp = grp.to(self.device).eval() if grp is not None else None
        self.uniform_init = uniform_init
        self.shared_stats = shared_stats  # {'count': Value, 'mean': Value, 'M2': Value, 'lock': Lock}

        pts = pts or [3, 1, -1, -3]
        self.pts = torch.tensor(pts, dtype=torch.float64, device=self.device)

    def calc_grp(self, grp_feature):
        seq = list(map(
            lambda idx: torch.as_tensor(grp_feature[:idx+1], device=self.device),
            range(len(grp_feature)),
        ))

        with torch.inference_mode():
            logits = self.grp(seq)
        matrix = self.grp.calc_matrix(logits)
        return matrix

    def calc_rank_prob(self, player_id, grp_feature, rank_by_player):
        matrix = self.calc_grp(grp_feature)

        final_ranking = torch.zeros((1, 4), device=self.device)
        final_ranking[0, rank_by_player[player_id]] = 1.
        rank_prob = torch.cat((matrix[:, player_id], final_ranking))
        if self.uniform_init:
            rank_prob[0, :] = 1 / 4
        return rank_prob

    def calc_delta_pt(self, player_id, grp_feature, rank_by_player):
        rank_prob = self.calc_rank_prob(player_id, grp_feature, rank_by_player)
        exp_pts = rank_prob @ self.pts
        reward = exp_pts[1:] - exp_pts[:-1]
        reward_np = reward.cpu().numpy()

        if self.shared_stats is not None:
            batch_count = len(reward_np)
            batch_mean = float(reward_np.mean())
            batch_M2 = float(((reward_np - batch_mean) ** 2).sum())

            with self.shared_stats['lock']:
                gc = self.shared_stats['count'].value
                gm = self.shared_stats['mean'].value
                gM2 = self.shared_stats['M2'].value

                # Use stats from BEFORE this batch for normalization
                if gc == 0:
                    std = 1.0
                    mean = 0.0
                else:
                    std = max(np.sqrt(gM2 / gc), 1e-8)
                    mean = gm

                # Chan's parallel formula to merge batch into global
                total = gc + batch_count
                delta = batch_mean - gm
                combined_mean = (gc * gm + batch_count * batch_mean) / total
                combined_M2 = gM2 + batch_M2 + delta ** 2 * gc * batch_count / total

                self.shared_stats['count'].value = total
                self.shared_stats['mean'].value = combined_mean
                self.shared_stats['M2'].value = combined_M2

            reward_np = (reward_np - mean) / std

        return reward_np

    def calc_delta_points(self, player_id, grp_feature, final_scores):
        seq = np.concatenate((grp_feature[:, 3 + player_id] * 1e4, [final_scores[player_id]]))
        delta_points = seq[1:] - seq[:-1]
        return delta_points
