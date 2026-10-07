import os
import dataclasses
import time
import logging
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from .evaluator import SemDisRecEvaluator, AttributionResults
from ..models.streams.stream_r import LeakageDiscriminator
from ..models.streams.stream_s import StreamS
from ..models.streams.behavioral import LBPN

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class SemDisRecTrainer:
    """
    SemDisRec++ Trainer implementing Algorithm 1: 3-phase forward pass + 2-phase adversarial mini-batch training.
    Phase 1: Update auditor discriminator phi to minimize semantic leakage L_leak.
    Phase 2: Update all main model parameters to minimize total 7-term loss objective L.
    """

    def __init__(self, model, dataset, config, device: torch.device, checkpoint_dir: str = None):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Sync feature dimension with dataset
        if dataset.p_u is not None and dataset.p_u.shape[1] != config.feature_dim:
            config.feature_dim = dataset.p_u.shape[1]
            model.sync_feature_dim(dataset.p_u.shape[1])

        self.model.init_embeddings(dataset.num_users, dataset.num_items)
        self.model.to(device)

        self.p_u = dataset.p_u.to(device)
        self.g_u = dataset.g_u.to(device)
        self.edge_index = dataset.pyg_data["user", "rates", "item"].edge_index.to(device)

        disc_params = list(self.model.discriminator.parameters())
        disc_param_ids = set(map(id, disc_params))
        main_params = [p for p in self.model.parameters() if id(p) not in disc_param_ids]

        self.opt_disc = Adam(disc_params, lr=config.lr, weight_decay=config.weight_decay)
        self.opt_main = Adam(main_params, lr=config.lr, weight_decay=config.weight_decay)

        self.evaluator = SemDisRecEvaluator(model, dataset, config, device)

    def _refresh_g_u(self):
        """
        Recomputes g_u with current ||e_u^R||_2 as column 1.
        Called at the start of every epoch (Sec 1.1.2, SemDisRec++).
        """
        self.dataset.update_g_u(self.model)
        self.g_u = self.dataset.g_u.to(self.device)

    def train_epoch(self) -> Dict[str, float]:
        """Runs 1 full epoch of 2-phase adversarial mini-batch training."""
        # Sec 1.1.2, SemDisRec++: refresh g_u with current ||e_u^R||_2 before every epoch.
        self._refresh_g_u()
        self.model.train()
        num_train_edges = len(self.dataset.train_src)
        perm = np.random.permutation(num_train_edges)

        train_src_arr = np.array(self.dataset.train_src)[perm]
        train_dst_arr = np.array(self.dataset.train_dst)[perm]

        batch_size = self.config.batch_size
        num_batches = int(np.ceil(num_train_edges / batch_size))

        total_loss_sum = 0.0
        link_loss_sum = 0.0
        orth_loss_sum = 0.0
        sem_loss_sum = 0.0
        ssl_loss_sum = 0.0
        leak_loss_sum = 0.0
        div_loss_sum = 0.0
        l2_loss_sum = 0.0

        for b in range(num_batches):
            start_idx = b * batch_size
            end_idx = min((b + 1) * batch_size, num_train_edges)

            batch_pos_u = torch.tensor(train_src_arr[start_idx:end_idx], dtype=torch.long, device=self.device)
            batch_pos_i = torch.tensor(train_dst_arr[start_idx:end_idx], dtype=torch.long, device=self.device)
            batch_neg_i = torch.randint(0, self.dataset.num_items, (len(batch_pos_u),), device=self.device)

            # Phase 1: Update Auditor Discriminator (freeze non-phi parameters)
            # [PERF] Only h_r1 is ever used here (h_r2 was previously computed via
            # forward_dual_view and immediately discarded) — forward_single_view
            # produces the same distribution for h_r1 (same noise generator, same
            # gate, same GNN encoder; z_u is freshly resampled either way), so this
            # removes one redundant full-graph GNN forward pass per batch with zero
            # change to the discriminator's loss, gradients, or update.
            self.opt_disc.zero_grad()
            with torch.no_grad():
                h_r1 = self.model.stream_r.forward_single_view(
                    self.model.user_emb_R,
                    self.model.item_emb_R,
                    self.g_u,
                    self.edge_index,
                    self.dataset.num_users,
                    self.dataset.num_items,
                )
            batch_users = torch.unique(batch_pos_u)
            # [FIX] Normalize h_r1 the same way compute_all_losses() does for the
            # adversarial term (F.normalize(..., dim=-1)) -- otherwise the
            # discriminator is trained on raw-scale vectors here but evaluated on
            # unit-norm vectors during the generator's adversarial update, which
            # are a different input distribution and miscalibrate the critic.
            p_hat = self.model.discriminator(torch.nn.functional.normalize(h_r1[batch_users], dim=-1))
            l_leak = torch.nn.functional.mse_loss(p_hat, self.p_u[batch_users], reduction="mean")
            l_leak.backward()
            self.opt_disc.step()

            # Phase 2: Update Main Model Parameters (freeze phi auditor)
            self.opt_main.zero_grad()
            losses = self.model.compute_all_losses(
                self.p_u,
                self.g_u,
                self.edge_index,
                batch_pos_u,
                batch_pos_i,
                batch_neg_i,
                self.dataset.user_train_items,
                self.dataset.num_users,
                self.dataset.num_items,
            )

            losses.total_loss.backward()
            self.opt_main.step()

            total_loss_sum += losses.total_loss.item()
            link_loss_sum += losses.link_loss.item()
            orth_loss_sum += losses.orth_loss.item()
            sem_loss_sum += losses.sem_loss.item()
            ssl_loss_sum += losses.ssl_loss.item()
            leak_loss_sum += losses.leak_loss.item()
            div_loss_sum += losses.div_loss.item()
            l2_loss_sum += losses.l2_loss.item()

        return {
            "total_loss": total_loss_sum / max(1, num_batches),
            "link_loss": link_loss_sum / max(1, num_batches),
            "orth_loss": orth_loss_sum / max(1, num_batches),
            "sem_loss": sem_loss_sum / max(1, num_batches),
            "ssl_loss": ssl_loss_sum / max(1, num_batches),
            "leak_loss": leak_loss_sum / max(1, num_batches),
            "div_loss": div_loss_sum / max(1, num_batches),
            "l2_loss": l2_loss_sum / max(1, num_batches),
        }

    def train(self) -> Tuple[Dict[str, float], AttributionResults]:
        """Full training loop with NDCG@20 early stopping and final attribution evaluation."""
        best_val_ndcg20 = -1.0
        best_epoch = 0
        best_weights = None
        patience_counter = 0

        logger.info("Starting SemDisRec++ training on dataset '%s' (device: %s)...", self.config.dataset_name, self.device)

        for epoch in range(1, self.config.max_epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch()
            t_epoch = time.time() - t0

            val_metrics = self.evaluator.evaluate(mode="val")
            val_ndcg20 = val_metrics.get("NDCG@20", 0.0)
            val_hr20 = val_metrics.get("HR@20", 0.0)
            val_ndcg10 = val_metrics.get("NDCG@10", 0.0)

            logger.info(
                "Epoch %03d [%.1fs] | Total: %.4f | BCE: %.4f | Orth: %.4f | Sem: %.4f | SSL^R: %.4f | Leak: %.4f | Div: %.4f | Val NDCG@20: %.4f | Val HR@20: %.4f | Val NDCG@10: %.4f",
                epoch, t_epoch, train_metrics["total_loss"], train_metrics["link_loss"],
                train_metrics["orth_loss"], train_metrics["sem_loss"], train_metrics["ssl_loss"],
                train_metrics["leak_loss"], train_metrics["div_loss"], val_ndcg20, val_hr20, val_ndcg10
            )

            if val_ndcg20 > best_val_ndcg20:
                best_val_ndcg20 = val_ndcg20
                best_epoch = epoch
                best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0

                if self.checkpoint_dir:
                    ckpt = {
                        "model_state_dict": best_weights,
                        "config": dataclasses.asdict(self.config),
                        "epoch": epoch,
                        "val_ndcg20": best_val_ndcg20,
                        "dataset_name": self.config.dataset_name
                    }
                    torch.save(ckpt, os.path.join(self.checkpoint_dir, "best_model.pt"))
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    logger.info("Early stopping triggered at epoch %d (best epoch: %d, Val NDCG@20: %.4f).", epoch, best_epoch, best_val_ndcg20)
                    break

        if best_weights is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_weights.items()})

        test_metrics = self.evaluator.evaluate(mode="test")
        attribution_results = self.evaluator.run_attribution_protocol()

        logger.info("========================================================")
        logger.info("Final Test Set Evaluation Results:")
        for k, v in test_metrics.items():
            logger.info("  %s: %.4f", k, v)
        logger.info("Stream Attribution Evaluation Protocol Results (HR@20 / NDCG@20 / MRR@20):")
        logger.info(
            "  Full Model:                       HR=%.4f NDCG=%.4f MRR=%.4f",
            attribution_results.full_hr20, attribution_results.full_ndcg20, attribution_results.full_mrr20,
        )
        logger.info(
            "  Stream R Zeroed (Stream S Only):  HR=%.4f NDCG=%.4f MRR=%.4f (Delta_S NDCG@20: %.4f)",
            attribution_results.r_zeroed_hr20, attribution_results.r_zeroed_ndcg20, attribution_results.r_zeroed_mrr20,
            attribution_results.delta_s,
        )
        logger.info(
            "  Stream S Zeroed (Stream R Only):  HR=%.4f NDCG=%.4f MRR=%.4f (Delta_R NDCG@20: %.4f)",
            attribution_results.s_zeroed_hr20, attribution_results.s_zeroed_ndcg20, attribution_results.s_zeroed_mrr20,
            attribution_results.delta_r,
        )
        logger.info(
            "  Feature-Shuffled Control:         HR=%.4f NDCG=%.4f MRR=%.4f",
            attribution_results.shuffled_hr20, attribution_results.shuffled_ndcg20, attribution_results.shuffled_mrr20,
        )
        logger.info("  Cold-Start (Sec 4.8.3, cohort=%d):", attribution_results.cold_start_cohort_size)
        logger.info("    Model (Stream S only):          %.4f", attribution_results.cold_start_ndcg)
        logger.info("    Random baseline:                %.4f", attribution_results.cold_start_random_ndcg)
        logger.info("    Popularity baseline:            %.4f", attribution_results.cold_start_popularity_ndcg)
        logger.info("========================================================")

        if self.checkpoint_dir:
            ckpt = {
                "model_state_dict": {k: v.cpu().clone() for k, v in self.model.state_dict().items()},
                "config": dataclasses.asdict(self.config),
                "epoch": best_epoch,
                "val_ndcg20": best_val_ndcg20,
                "dataset_name": self.config.dataset_name,
                "test_metrics": test_metrics
            }
            torch.save(ckpt, os.path.join(self.checkpoint_dir, "final_model.pt"))

        return test_metrics, attribution_results

def load_checkpoint(checkpoint_path: str, device: torch.device) -> dict:
    return torch.load(checkpoint_path, map_location=device, weights_only=False)
