import logging

import numpy as np

import torch
import torch.nn as nn

import lightning.pytorch
import lightning.pytorch as pl

from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

# ---------------------------
# Metrics
# ---------------------------

spearman_scorer = make_scorer(
    lambda y_true, y_pred: spearmanr(y_true, y_pred).statistic,
    greater_is_better=True,
)


def spearman_perm_test(y_true, y_pred, n_resamples=1000):
    def spearman_stat(x, y):
        return spearmanr(x, y)[0]

    observed = spearman_stat(y_true, y_pred)

    null = np.zeros(n_resamples)
    for i in range(n_resamples):
        null[i] = spearman_stat(y_true, np.random.permutation(y_pred))

    p = (np.sum(np.abs(null) >= abs(observed)) + 1) / (n_resamples + 1)
    return p


# ---------------------------
# Helpers
# ---------------------------


def _train_best_model_on_fold(X_train, X_test, y_train, y_test, n_inits):
    best_r = -np.inf
    best_model = None
    best_pred_test = None

    for _ in range(n_inits):
        model = Pipeline(
            [
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=(10,),
                        learning_rate_init=0.0005,
                        alpha=0.0005,
                        max_iter=2000,
                        tol=0.001,
                        n_iter_no_change=10,
                    ),
                )
            ]
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r = spearmanr(y_test, y_pred).statistic

        if r > best_r:
            best_r = r
            best_model = model
            best_pred_test = y_pred

    return best_model, best_pred_test


# ---------------------------
# Main function
# ---------------------------


def MLP_regression(
    hne_feats,
    bio_feats,
    cv_splits=5,
    n_inits=5,
):
    X = hne_feats.X

    corrs = {}
    preds = {}
    pvals = {}

    crossval = KFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=42,
    )

    for col in tqdm(sorted(bio_feats.var_names), desc="Targets"):
        y = bio_feats[:, col].X.ravel()

        y_pred_train, y_true_train = [], []
        y_pred_test, y_true_test = [], []

        for train_idx, test_idx in tqdm(
            crossval.split(X),
            total=cv_splits,
            leave=False,
            desc=f"{col} folds",
        ):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # model selection
            best_model, y_pred_test_cv = _train_best_model_on_fold(
                X_train, X_test, y_train, y_test, n_inits
            )

            # predictions
            y_pred_train_cv = best_model.predict(X_train)

            y_pred_train.append(y_pred_train_cv)
            y_true_train.append(y_train)
            y_pred_test.append(y_pred_test_cv)
            y_true_test.append(y_test)

        # aggregate predictions
        y_true_test_all = np.concatenate(y_true_test)
        y_pred_test_all = np.concatenate(y_pred_test)

        y_true_train_all = np.concatenate(y_true_train)
        y_pred_train_all = np.concatenate(y_pred_train)

        test_r = spearmanr(y_true_test_all, y_pred_test_all).statistic
        train_r = spearmanr(y_true_train_all, y_pred_train_all).statistic

        corrs[col] = {
            "test": test_r,
            "train": train_r,
        }

        pvals[col] = spearman_perm_test(
            y_true_test_all,
            y_pred_test_all,
        )

        preds[col] = {
            "y_true_test": y_true_test_all,
            "y_pred_test": y_pred_test_all,
        }

        # if np.abs(test_r) >= shap_corr_threshold:
        #     shap_values_all[col] = compute_cv_shap(
        #         fold_cache=fold_cache,
        #         shap_topk=shap_topk,
        #     )
        # else:
        #     shap_values_all[col] = None

    return corrs, preds, pvals  # , shap_values_all


# # =========================
# # MODEL
# # =========================
# class LightningMLP(pl.LightningModule):
#     def __init__(
#         self,
#         input_dim,
#         hidden_dims=(10,),
#         lr=5e-4,
#         weight_decay=1e-4,
#     ):
#         super().__init__()
#         self.save_hyperparameters()

#         layers = []
#         prev = input_dim
#         for h in hidden_dims:
#             layers.append(nn.Linear(prev, h))
#             layers.append(nn.Tanh())
#             prev = h
#         layers.append(nn.Linear(prev, 1))

#         self.net = nn.Sequential(*layers)
#         self.loss_fn = nn.MSELoss()

#     def forward(self, x):
#         return self.net(x)

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         y = y.unsqueeze(-1)
#         y_hat = self(x)
#         loss = self.loss_fn(y_hat, y)
#         return loss

#     def configure_optimizers(self):
#         return torch.optim.Adam(
#             self.parameters(),
#             lr=self.hparams.lr,
#             weight_decay=self.hparams.weight_decay,
#         )


# # =========================
# # WRAPPER
# # =========================
# class LightningMLPWrapper:
#     def __init__(
#         self,
#         input_dim,
#         hidden_dims,
#         lr=5e-4,
#         weight_decay=1e-4,
#         max_epochs=200,
#         batch_size=64,
#         seed=None,
#         device="cpu",
#     ):
#         if seed is not None:
#             torch.manual_seed(seed)
#             np.random.seed(seed)

#         self.model = LightningMLP(
#             input_dim=input_dim,
#             hidden_dims=hidden_dims,
#             lr=lr,
#             weight_decay=weight_decay,
#         )

#         self.trainer = pl.Trainer(
#             max_epochs=max_epochs,
#             accelerator="cpu",
#             devices=1,
#             logger=False,
#             enable_checkpointing=False,
#             enable_progress_bar=False,
#             enable_model_summary=False,
#         )

#         self.batch_size = batch_size

#     def fit(self, X, y):
#         X_t = torch.tensor(X, dtype=torch.float32)
#         y_t = torch.tensor(y, dtype=torch.float32)

#         ds = TensorDataset(X_t, y_t)
#         dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

#         self.trainer.fit(self.model, dl)
#         return self

#     def predict(self, X):
#         self.model.eval()
#         X_t = torch.tensor(X, dtype=torch.float32)

#         dl = DataLoader(X_t, batch_size=1024)
#         preds = []

#         with torch.no_grad():
#             for xb in dl:
#                 preds.append(self.model(xb).numpy())

#         return np.concatenate(preds).squeeze(-1)


# # =========================
# # TRAIN BEST MODEL (RESTARTS)
# # =========================
# def _train_best_model_on_fold(
#     X_train,
#     X_test,
#     y_train,
#     y_test,
#     n_inits,
# ):
#     best_r = -np.inf
#     best_model = None
#     best_pred_test = None

#     input_dim = X_train.shape[1]

#     for i in range(n_inits):
#         model = LightningMLPWrapper(
#             input_dim=input_dim,
#             hidden_dims=(10,),
#             lr=5e-4,
#             weight_decay=1e-4,
#             max_epochs=200,
#             batch_size=64,
#             seed=i,
#             device="cpu",
#         )

#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)

#         r = spearmanr(y_test, y_pred).statistic

#         if r > best_r:
#             best_r = r
#             best_model = model
#             best_pred_test = y_pred

#     return best_model, best_pred_test


# # =========================
# # PARALLEL FOLD TRAINING
# # =========================
# def _configure_worker_logging():
#     import os
#     import warnings
#     import logging
#     warnings.filterwarnings("ignore")

#     logging.getLogger().handlers.clear()
#     logging.basicConfig(level=logging.CRITICAL)
#     logging.getLogger().setLevel(logging.CRITICAL)

#     logging.getLogger("lightning.pytorch").setLevel(logging.CRITICAL)
#     logging.getLogger("lightning").setLevel(logging.CRITICAL)
#     logging.getLogger("torch").setLevel(logging.CRITICAL)
#     logging.getLogger("joblib").setLevel(logging.CRITICAL)
#     logging.getLogger("shap").setLevel(logging.CRITICAL)

# def _train_one_fold(
#     X,
#     y,
#     train_idx,
#     test_idx,
#     n_inits,
# ):
#     _configure_worker_logging()

#     X_train, X_test = X[train_idx], X[test_idx]
#     y_train, y_test = y[train_idx], y[test_idx]

#     best_model, y_pred_test = _train_best_model_on_fold(
#         X_train, X_test, y_train, y_test, n_inits
#     )

#     y_pred_train = best_model.predict(X_train)

#     return {
#         "model": best_model,
#         "X_train": X_train,
#         "X_test": X_test,
#         "y_train": y_train,
#         "y_test": y_test,
#         "y_pred_train": y_pred_train,
#         "y_pred_test": y_pred_test,
#     }


# # =========================
# # SHAP
# # =========================
# def _compute_fold_shap_full(
#     model_wrapper,
#     X_train,
#     X_test,
#     background_size=64,
# ):
#     model = model_wrapper.model
#     model.eval()

#     if X_train.shape[0] > background_size:
#         idx = np.random.choice(X_train.shape[0], background_size, replace=False)
#         bg = X_train[idx]
#     else:
#         bg = X_train

#     bg_t = torch.tensor(bg, dtype=torch.float32)
#     X_test_t = torch.tensor(X_test, dtype=torch.float32)

#     explainer = shap.GradientExplainer(model, bg_t)
#     shap_vals = explainer.shap_values(X_test_t)

#     if isinstance(shap_vals, list):
#         shap_vals = shap_vals[0]

#     return shap_vals


# def compute_cv_shap(
#     fold_cache,
#     background_size=64,
# ):
#     shap_values = []

#     for fold in fold_cache:
#         shap_vals = _compute_fold_shap_full(
#             model_wrapper=fold["model"],
#             X_train=fold["X_train"],
#             X_test=fold["X_test"],
#             background_size=background_size,
#         )
#         shap_values.append(shap_vals)

#     shap_concat = np.concatenate(shap_values, axis=0)

#     return {
#         "values": shap_concat,
#         "importance": {
#             "mean_abs": np.mean(np.abs(shap_concat), axis=0),
#             "median_abs": np.median(np.abs(shap_concat), axis=0),
#         },
#     }


# # =========================
# # MAIN FUNCTION (PARALLEL CV)
# # =========================
# def MLP_regression(
#     adata,
#     exp_columns,
#     cv_splits=5,
#     n_inits=5,
#     shap_corr_threshold=0.5,
#     background_size=64,
# ):
#     X = adata.X

#     corrs = {}
#     preds = {}
#     pvals = {}
#     shap_values_all = {}

#     crossval = KFold(
#         n_splits=cv_splits,
#         shuffle=True,
#         random_state=42,
#     )

#     for col in tqdm(sorted(exp_columns), desc="Targets"):
#         y = adata.obs[col].values

#         fold_results = Parallel(n_jobs=cv_splits)(
#             delayed(_train_one_fold)(
#                 X,
#                 y,
#                 train_idx,
#                 test_idx,
#                 n_inits,
#             )
#             for train_idx, test_idx in crossval.split(X)
#         )

#         y_true_test_all = np.concatenate([r["y_test"] for r in fold_results])
#         y_pred_test_all = np.concatenate([r["y_pred_test"] for r in fold_results])

#         y_true_train_all = np.concatenate([r["y_train"] for r in fold_results])
#         y_pred_train_all = np.concatenate([r["y_pred_train"] for r in fold_results])

#         test_r = spearmanr(y_true_test_all, y_pred_test_all).statistic
#         train_r = spearmanr(y_true_train_all, y_pred_train_all).statistic

#         corrs[col] = {"test": test_r, "train": train_r}

#         pvals[col] = spearman_perm_test(
#             y_true_test_all,
#             y_pred_test_all,
#         )

#         preds[col] = {
#             "y_true_test": y_true_test_all,
#             "y_pred_test": y_pred_test_all,
#         }

#         fold_cache = [
#             {
#                 "model": r["model"],
#                 "X_train": r["X_train"],
#                 "X_test": r["X_test"],
#             }
#             for r in fold_results
#         ]

#         if abs(test_r) >= shap_corr_threshold:
#             shap_values_all[col] = compute_cv_shap(
#                 fold_cache=fold_cache,
#                 background_size=background_size,
#             )
#         else:
#             shap_values_all[col] = None

#     return corrs, preds, pvals, shap_values_all
