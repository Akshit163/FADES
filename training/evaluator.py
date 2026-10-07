from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class AttributionResults:
    full_metrics: Dict[str, float]
    stream_r_zeroed: Dict[str, float]
    stream_s_zeroed: Dict[str, float]
    shuffled_features: Dict[str, float]
    delta_r: float
    delta_s: float
    full_ndcg20: float = 0.0
    r_zeroed_ndcg20: float = 0.0
    s_zeroed_ndcg20: float = 0.0
    shuffled_ndcg20: float = 0.0
    full_hr20: float = 0.0
    r_zeroed_hr20: float = 0.0
    s_zeroed_hr20: float = 0.0
    shuffled_hr20: float = 0.0
    full_mrr20: float = 0.0
    r_zeroed_mrr20: float = 0.0
    s_zeroed_mrr20: float = 0.0
    shuffled_mrr20: float = 0.0
    cold_start_cohort_size: int = 0
    cold_start_ndcg: float = 0.0
    cold_start_random_ndcg: float = 0.0
    cold_start_popularity_ndcg: float = 0.0


def compute_ranking_metrics(scores_matrix: np.ndarray, pos_indices: List[int], k_list: List[int] = [10, 20]) -> Dict[str, float]:
    num_users = len(scores_matrix)
    metrics = {}
    
    # ADD NOISE ONCE GLOBALLY FOR FAST TIE-BREAKING
    noise = np.random.uniform(-1e-9, 1e-9, size=scores_matrix.shape)
    scores_matrix = scores_matrix + noise

    for k in k_list:
        hr_sum, ndcg_sum, recall_sum, prec_sum, mrr_sum = 0.0, 0.0, 0.0, 0.0, 0.0
        for u in range(num_users):
            user_scores = scores_matrix[u]
            pos_idx = pos_indices[u]
            pos_score = user_scores[pos_idx]
            
            # Rank calculated strictly > + 1 since noise breaks exact ties
            rank = int(np.sum(user_scores > pos_score)) + 1

            if rank <= k:
                hr_sum += 1.0
                ndcg_sum += 1.0 / np.log2(rank + 1)
                recall_sum += 1.0
                prec_sum += 1.0 / k
                mrr_sum += 1.0 / rank

        metrics[f"HR@{k}"] = hr_sum / max(1, num_users)
        metrics[f"NDCG@{k}"] = ndcg_sum / max(1, num_users)
        metrics[f"Recall@{k}"] = recall_sum / max(1, num_users)
        metrics[f"Precision@{k}"] = prec_sum / max(1, num_users)
        metrics[f"MRR@{k}"] = mrr_sum / max(1, num_users)

    return metrics


class SemDisRecEvaluator:
    def __init__(self, model, dataset, config, device: torch.device):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = device

        if dataset.p_u is not None:
            model.sync_feature_dim(dataset.p_u.shape[1])

        self.p_u = dataset.p_u.to(device)
        self.g_u = dataset.g_u.to(device)
        self.edge_index = dataset.pyg_data["user", "rates", "item"].edge_index.to(device)

    def evaluate(
        self,
        mode: str = "test",
        zero_r: bool = False,
        zero_s: bool = False,
        shuffle_p_u: bool = False,
        eval_protocol: Optional[str] = None,
    ) -> Dict[str, float]:
        protocol = eval_protocol if eval_protocol is not None else getattr(self.config, "eval_protocol", "full")
        if protocol == "full":
            return self.evaluate_full(mode=mode, zero_r=zero_r, zero_s=zero_s, shuffle_p_u=shuffle_p_u)
        else:
            return self.evaluate_sampled(mode=mode, zero_r=zero_r, zero_s=zero_s, shuffle_p_u=shuffle_p_u)

    def evaluate_full(
        self,
        mode: str = "test",
        zero_r: bool = False,
        zero_s: bool = False,
        shuffle_p_u: bool = False,
    ) -> Dict[str, float]:
        self.model.eval()
        target_dict = self.dataset.test_dict if mode == "test" else self.dataset.val_dict
        other_dict = self.dataset.val_dict if mode == "test" else self.dataset.test_dict
        eval_users = [u for u in target_dict if len(target_dict[u]) > 0]

        if len(eval_users) == 0:
            return {
                f"{name}@{k}": 0.0
                for k in self.config.k_list
                for name in ("HR", "NDCG", "Recall", "Precision", "MRR")
            }

        with torch.no_grad():
            h_user, e_item, _, _ = self.model.forward_representation(
                self.p_u,
                self.g_u,
                self.edge_index,
                self.dataset.num_users,
                self.dataset.num_items,
                zero_r=zero_r,
                zero_s=zero_s,
                shuffle_p_u=shuffle_p_u,
            )
            full_scores = torch.matmul(h_user, e_item.t())

            for u in range(self.dataset.num_users):
                train_items = self.dataset.user_train_items.get(u, [])
                if train_items:
                    full_scores[u, train_items] = -1e9

            full_scores_np = full_scores.cpu().numpy()
            
            # FAST FIX: Add infinitesimal noise ONCE to the entire matrix
            noise = np.random.uniform(-1e-9, 1e-9, size=full_scores_np.shape)
            full_scores_np = full_scores_np + noise

        metrics = {}
        for k in self.config.k_list:
            hr_sum, ndcg_sum, recall_sum, prec_sum, mrr_sum = 0.0, 0.0, 0.0, 0.0, 0.0
            for u in eval_users:
                pos_items = target_dict[u]
                cross_items = other_dict.get(u, [])

                user_scores = full_scores_np[u]
                if cross_items:
                    user_scores = user_scores.copy()
                    user_scores[cross_items] = -1e9

                ranks = []
                for pos_item in pos_items:
                    siblings = [it for it in pos_items if it != pos_item]
                    scores_for_item = user_scores
                    if siblings:
                        scores_for_item = user_scores.copy()
                        scores_for_item[siblings] = -1e9

                    pos_score = scores_for_item[pos_item]
                    # No slow noise generation here anymore!
                    rank = int(np.sum(scores_for_item > pos_score)) + 1
                    ranks.append(rank)

                num_pos = len(pos_items)
                hit_ranks = [r for r in ranks if r <= k]
                num_hits = len(hit_ranks)

                if num_hits > 0:
                    hr_sum += 1.0
                    dcg = sum(1.0 / np.log2(r + 1) for r in hit_ranks)
                    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(num_pos, k)))
                    ndcg_sum += (dcg / idcg) if idcg > 0 else 0.0
                    recall_sum += num_hits / num_pos
                    prec_sum += num_hits / k
                    mrr_sum += 1.0 / min(ranks)

            num_u = max(1, len(eval_users))
            metrics[f"HR@{k}"] = hr_sum / num_u
            metrics[f"NDCG@{k}"] = ndcg_sum / num_u
            metrics[f"Recall@{k}"] = recall_sum / num_u
            metrics[f"Precision@{k}"] = prec_sum / num_u
            metrics[f"MRR@{k}"] = mrr_sum / num_u

        return metrics

    def evaluate_sampled(
        self,
        mode: str = "test",
        zero_r: bool = False,
        zero_s: bool = False,
        shuffle_p_u: bool = False,
    ) -> Dict[str, float]:
        self.model.eval()
        target_dict = self.dataset.test_dict if mode == "test" else self.dataset.val_dict
        eval_users = [u for u in target_dict if len(target_dict[u]) > 0]

        if len(eval_users) == 0:
            return {f"NDCG@{k}": 0.0 for k in self.config.k_list}

        with torch.no_grad():
            h_user, e_item, _, _ = self.model.forward_representation(
                self.p_u,
                self.g_u,
                self.edge_index,
                self.dataset.num_users,
                self.dataset.num_items,
                zero_r=zero_r,
                zero_s=zero_s,
                shuffle_p_u=shuffle_p_u,
            )

        rng = np.random.default_rng(self.config.seed)
        scores_matrix = []

        for u in eval_users:
            pos_item = target_dict[u][0]
            train_items = set(self.dataset.user_train_items.get(u, []))
            avail_negs = [i for i in range(self.dataset.num_items) if i not in train_items and i != pos_item]
            if len(avail_negs) >= self.config.num_negatives:
                neg_items = list(rng.choice(avail_negs, size=self.config.num_negatives, replace=False))
            elif len(avail_negs) > 0:
                neg_items = list(rng.choice(avail_negs, size=self.config.num_negatives, replace=True))
            else:
                neg_items = [pos_item] * self.config.num_negatives

            candidate_items = [pos_item] + neg_items
            items_tensor = torch.tensor(candidate_items, dtype=torch.long, device=self.device)
            u_tensor = torch.tensor([u] * len(candidate_items), dtype=torch.long, device=self.device)

            with torch.no_grad():
                scores = self.model.predict(u_tensor, items_tensor, h_user, e_item).cpu().numpy()

            scores_matrix.append(scores)

        scores_matrix = np.array(scores_matrix)
        return compute_ranking_metrics(scores_matrix, pos_indices=[0] * len(eval_users), k_list=self.config.k_list)

    def evaluate_cold_start(self) -> Dict[str, float]:
        target_dict = self.dataset.test_dict
        cold_users = [u for u in target_dict if len(self.dataset.user_train_items.get(u, [])) == 0 and len(target_dict[u]) > 0]

        if len(cold_users) == 0:
            return {
                "model_ndcg@10": 0.0,
                "random_ndcg@10": 0.0,
                "popularity_ndcg@10": 0.0,
                "cohort_size": 0,
            }

        self.model.eval()
        with torch.no_grad():
            b_u_cold = self.model.lbpn.cold_start_mlp(self.p_u)
            h_u_s = torch.matmul(b_u_cold, self.model.lbpn.h_s_proj.weight)
            h_r_zero = torch.zeros_like(h_u_s)
            h_user = torch.cat([h_r_zero, h_u_s], dim=-1)
            e_item = torch.cat([self.model.item_emb_R, self.model.item_emb_S], dim=-1)
            
            full_scores = torch.matmul(h_user, e_item.t()).cpu().numpy()
            # FAST FIX: Add noise globally
            noise_m = np.random.uniform(-1e-9, 1e-9, size=full_scores.shape)
            full_scores = full_scores + noise_m

        rng = np.random.default_rng(self.config.seed)
        model_ndcg_sum, random_ndcg_sum, pop_ndcg_sum = 0.0, 0.0, 0.0

        item_counts = defaultdict(int)
        for src_i in self.dataset.train_dst:
            item_counts[src_i] += 1
            
        pop_scores = np.array([item_counts[i] for i in range(self.dataset.num_items)], dtype=np.float32)
        # FAST FIX: Add noise globally
        noise_p = np.random.uniform(-1e-9, 1e-9, size=pop_scores.shape)
        pop_scores = pop_scores + noise_p

        for u in cold_users:
            pos_item = target_dict[u][0]

            u_scores = full_scores[u]
            pos_score = u_scores[pos_item]
            m_rank = int(np.sum(u_scores > pos_score)) + 1
            if m_rank <= 10:
                model_ndcg_sum += 1.0 / np.log2(m_rank + 1)

            r_rank = int(rng.integers(1, self.dataset.num_items + 1))
            if r_rank <= 10:
                random_ndcg_sum += 1.0 / np.log2(r_rank + 1)

            p_score = pop_scores[pos_item]
            p_rank = int(np.sum(pop_scores > p_score)) + 1
            if p_rank <= 10:
                pop_ndcg_sum += 1.0 / np.log2(p_rank + 1)

        num_cold = len(cold_users)
        return {
            "model_ndcg@10": model_ndcg_sum / num_cold,
            "random_ndcg@10": random_ndcg_sum / num_cold,
            "popularity_ndcg@10": pop_ndcg_sum / num_cold,
            "cohort_size": num_cold,
        }

    def run_attribution_protocol(self) -> AttributionResults:
        full_metrics = self.evaluate(mode="test")
        r_zeroed_metrics = self.evaluate(mode="test", zero_r=True)
        s_zeroed_metrics = self.evaluate(mode="test", zero_s=True)
        shuffled_metrics = self.evaluate(mode="test", shuffle_p_u=True)
        cold_start_res = self.evaluate_cold_start()

        full_ndcg20 = full_metrics.get("NDCG@20", 0.0)
        r_zeroed_ndcg20 = r_zeroed_metrics.get("NDCG@20", 0.0)
        s_zeroed_ndcg20 = s_zeroed_metrics.get("NDCG@20", 0.0)
        shuffled_ndcg20 = shuffled_metrics.get("NDCG@20", 0.0)

        delta_r = full_ndcg20 - r_zeroed_ndcg20
        delta_s = full_ndcg20 - s_zeroed_ndcg20

        return AttributionResults(
            full_metrics=full_metrics,
            stream_r_zeroed=r_zeroed_metrics,
            stream_s_zeroed=s_zeroed_metrics,
            shuffled_features=shuffled_metrics,
            delta_r=delta_r,
            delta_s=delta_s,
            full_ndcg20=full_ndcg20,
            r_zeroed_ndcg20=r_zeroed_ndcg20,
            s_zeroed_ndcg20=s_zeroed_ndcg20,
            shuffled_ndcg20=shuffled_ndcg20,
            full_hr20=full_metrics.get("HR@20", 0.0),
            r_zeroed_hr20=r_zeroed_metrics.get("HR@20", 0.0),
            s_zeroed_hr20=s_zeroed_metrics.get("HR@20", 0.0),
            shuffled_hr20=shuffled_metrics.get("HR@20", 0.0),
            full_mrr20=full_metrics.get("MRR@20", 0.0),
            r_zeroed_mrr20=r_zeroed_metrics.get("MRR@20", 0.0),
            s_zeroed_mrr20=s_zeroed_metrics.get("MRR@20", 0.0),
            shuffled_mrr20=shuffled_metrics.get("MRR@20", 0.0),
            cold_start_cohort_size=int(cold_start_res.get("cohort_size", 0)),
            cold_start_ndcg=float(cold_start_res.get("model_ndcg@10", 0.0)),
            cold_start_random_ndcg=float(cold_start_res.get("random_ndcg@10", 0.0)),
            cold_start_popularity_ndcg=float(cold_start_res.get("popularity_ndcg@10", 0.0)),
        )