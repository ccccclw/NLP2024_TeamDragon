from dataclasses import dataclass
import torch

@dataclass
class BeamSearch:
    b_size: int
    candidates: list
    scores: list

    def iter(self, curr_prob, prev_beamS, is_done):

        prev_scores = torch.tensor(prev_beamS.scores, dtype=curr_prob.dtype, device=curr_prob.device)

        # shape (batch_size, vocab_size)
        scores = curr_prob + prev_scores.unsqueeze(-1).expand_as(curr_prob)

        best_scores, best_idx = torch.sort(scores.view(-1))
        best_scores = best_scores[:self.b_size]
        best_idx = best_idx[:self.b_size]

        token_idx = best_idx - (best_idx / curr_prob.size(1)) * curr_prob.size(1)

        res_list, remain_list = [], []

        for b_score, b_idx, t_idx in zip(best_scores.tolist(), best_idx.tolist(), token_idx.tolist()):

            candidate = prev_beamS.candidates[b_idx] + [t_idx]

            if is_done(candidate):

                res_list.append([candidate, b_score])

            else:

                remain_list.append(b_idx)
                self.candidates.append(candidate)
                self.scores.append(b_score)

        return res_list, remain_list