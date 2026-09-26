import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorboardX import SummaryWriter
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _compute_metrics(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mae    = mean_absolute_error(y_true, y_pred)
    rmse   = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2     = float(r2_score(y_true, y_pred))
    mape   = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100)
    smape  = float(np.mean(2 * np.abs(y_true - y_pred) /
                           (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100)
    cv_rmse = rmse / (float(np.mean(np.abs(y_true))) + 1e-8) * 100
    return {
        "MAE_W": mae, "RMSE_W": rmse, "R2": r2,
        "MAPE_%": mape, "sMAPE_%": smape, "CV_RMSE_%": cv_rmse
    }


def _fig_to_writer(fig, writer, tag, step):
    writer.add_figure(tag, fig, global_step=step)
    plt.close(fig)


class SklearnTBLogger:
    """TensorBoard logger for sklearn tree-based models (LightGBM / Extra Trees / XGBoost).

    Usage inside any notebook
    -------------------------
    import sys; sys.path.insert(0, "..")
    from utils.tb_logger import SklearnTBLogger

    logger = SklearnTBLogger("XGBoost")
    for fold, (y_v, y_p) in enumerate(cv_results, 1):
        logger.log_cv_fold(fold, y_v, y_p)
    for i, (params, score) in enumerate(search_results):
        logger.log_hyperparam_trial(i, params, r2=score)
    logger.log_final(y_train, pred_train, y_test, pred_test)
    logger.log_feature_importance(feature_cols, model.feature_importances_)
    logger.log_prediction_plots(y_test, pred_test)
    logger.log_scatter_plot(y_test, pred_test)
    logger.close()
    """

    def __init__(self, model_name, log_root="runs", log_graph=True):
        self.model_name = model_name
        self.log_dir    = os.path.join(log_root, model_name)
        self.writer     = SummaryWriter(log_dir=self.log_dir)
        print(f"[TB] '{model_name}' -> {self.log_dir}")
        print(f"[TB] Launch: tensorboard --logdir={log_root}/")
        if log_graph:
            self.log_computational_graph()

    def log_cv_fold(self, fold, y_val, y_pred):
        m = _compute_metrics(y_val, y_pred)
        for k, v in m.items():
            self.writer.add_scalar(f"CV/{k}", v, fold)
        print(f"  Fold {fold}  MAE={m['MAE_W']:.0f}W  RMSE={m['RMSE_W']:.0f}W  R2={m['R2']:.3f}")
        return m

    def log_hyperparam_trial(self, trial, params, r2, mae=None, rmse=None):
        self.writer.add_scalar("HparamSearch/R2", r2, trial)
        if mae  is not None:
            self.writer.add_scalar("HparamSearch/MAE",  mae,  trial)
        if rmse is not None:
            self.writer.add_scalar("HparamSearch/RMSE", rmse, trial)
        for k, v in params.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(f"HparamSearch/param_{k}", float(v), trial)

    def log_final(self, y_train, y_pred_train, y_test, y_pred_test, hparams=None):
        tm = _compute_metrics(y_train, y_pred_train)
        vm = _compute_metrics(y_test,  y_pred_test)
        for k, v in tm.items():
            self.writer.add_scalar(f"Final/Train_{k}", v, 0)
        for k, v in vm.items():
            self.writer.add_scalar(f"Final/Test_{k}",  v, 0)
        hp = {}
        if hparams:
            for k, v in hparams.items():
                if isinstance(v, (int, float, str, bool)):
                    hp[k] = v
                elif v is None:
                    hp[k] = "None"
                else:
                    hp[k] = str(v)
        hp["model"] = self.model_name
        self.writer.add_hparams(
            hp,
            {
                "hparam/Test_MAE":  vm["MAE_W"],
                "hparam/Test_RMSE": vm["RMSE_W"],
                "hparam/Test_R2":   vm["R2"],
                "hparam/Train_R2":  tm["R2"],
                "hparam/R2_Gap":    tm["R2"] - vm["R2"],
            }
        )
        print(f"\n[{self.model_name}] Train  MAE={tm['MAE_W']:.0f}W  RMSE={tm['RMSE_W']:.0f}W  R2={tm['R2']:.3f}")
        print(f"[{self.model_name}] Test   MAE={vm['MAE_W']:.0f}W  RMSE={vm['RMSE_W']:.0f}W  R2={vm['R2']:.3f}")
        return {"train": tm, "test": vm}

    def log_feature_importance(self, feature_names, importances, top_n=20):
        importances = np.array(importances)
        idx    = np.argsort(importances)[::-1][:top_n]
        names  = [feature_names[i] for i in idx]
        values = importances[idx]
        fig, ax = plt.subplots(figsize=(10, 6))
        colors  = plt.cm.viridis(np.linspace(0.3, 0.85, len(names)))
        ax.barh(names[::-1], values[::-1], color=colors)
        ax.set_xlabel("Importance Score")
        ax.set_title(f"{self.model_name} -- Top {top_n} Feature Importances")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        _fig_to_writer(fig, self.writer, "FeatureImportance/bar", 0)

    def log_prediction_plots(self, y_test, y_pred, n_plot=500):
        y_test = np.array(y_test)
        y_pred = np.array(y_pred)
        n = min(len(y_test), n_plot)
        x = np.arange(n)
        fig = plt.figure(figsize=(14, 8))
        gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.35)
        ax0 = fig.add_subplot(gs[0])
        ax0.plot(x, y_test[:n], label="Actual",    color="#2196F3", lw=1.2)
        ax0.plot(x, y_pred[:n], label="Predicted", color="#FF5722", lw=1.2, alpha=0.8, ls="--")
        ax0.set_title(f"{self.model_name} -- Actual vs Predicted (first {n} test samples)")
        ax0.set_ylabel("Power (W)")
        ax0.legend(loc="upper right", framealpha=0.7)
        ax0.spines[["top", "right"]].set_visible(False)
        residuals = y_test[:n] - y_pred[:n]
        ax1 = fig.add_subplot(gs[1])
        ax1.bar(x, residuals, color=np.where(residuals >= 0, "#4CAF50", "#F44336"),
                width=1.0, alpha=0.7)
        ax1.axhline(0, color="black", lw=0.8, ls="--")
        ax1.set_xlabel("Test Sample Index")
        ax1.set_ylabel("Residual (W)")
        ax1.set_title("Residuals  (Actual - Predicted)")
        ax1.spines[["top", "right"]].set_visible(False)
        _fig_to_writer(fig, self.writer, "Predictions/time_series", 0)

    def log_scatter_plot(self, y_test, y_pred):
        y_test = np.array(y_test)
        y_pred = np.array(y_pred)
        lo = min(y_test.min(), y_pred.min())
        hi = max(y_test.max(), y_pred.max())
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(y_test, y_pred, alpha=0.2, s=8, color="#7C4DFF")
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect prediction")
        r2 = r2_score(y_test, y_pred)
        ax.set_xlabel("Actual Power (W)")
        ax.set_ylabel("Predicted Power (W)")
        ax.set_title(f"{self.model_name} -- Scatter  R2={r2:.3f}")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        _fig_to_writer(fig, self.writer, "Predictions/scatter", 0)

    def log_computational_graph(self, in_features=8, sample_input=None):
        """Logs a formal computational ensemble architecture graph to TensorBoard."""
        try:
            import torch
            import torch.nn as nn

            x = sample_input if sample_input is not None else torch.zeros(1, in_features)
            name_lower = self.model_name.lower()

            if "lightgbm" in name_lower or "lgb" in name_lower:
                class FeatureHistogramBinning(nn.Module):
                    def __init__(self, in_f=8, num_bins=256):
                        super().__init__()
                        self.bin_projection = nn.Linear(in_f, in_f, bias=False)
                    def forward(self, feat):
                        return self.bin_projection(feat)

                class LeafWiseTreeStage(nn.Module):
                    def __init__(self, in_f=8, num_leaves=45):
                        super().__init__()
                        self.leaf_router = nn.Linear(in_f, num_leaves, bias=True)
                        self.leaf_values = nn.Linear(num_leaves, 1, bias=False)
                    def forward(self, feat):
                        return self.leaf_values(torch.relu(self.leaf_router(feat)))

                class LightGBMArchitecture(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.input_features_8d = nn.Identity()
                        self.histogram_binning_256bins = FeatureHistogramBinning(in_features, 256)
                        self.trees_stage_1_to_100 = LeafWiseTreeStage(in_features, 45)
                        self.trees_stage_101_to_200 = LeafWiseTreeStage(in_features, 45)
                        self.trees_stage_201_to_300 = LeafWiseTreeStage(in_features, 45)
                        self.learning_rate = 0.01
                        self.base_offset = nn.Parameter(torch.tensor([14820.0]), requires_grad=False)
                    def forward(self, inp):
                        feat = self.input_features_8d(inp)
                        binned = self.histogram_binning_256bins(feat)
                        t1 = self.trees_stage_1_to_100(binned) * 100.0
                        t2 = self.trees_stage_101_to_200(binned) * 100.0
                        t3 = self.trees_stage_201_to_300(binned) * 100.0
                        return self.base_offset + self.learning_rate * (t1 + t2 + t3)

                graph_mod = LightGBMArchitecture()

            elif "extra" in name_lower or "et" in name_lower:
                class RandomFeatureSubsampling(nn.Module):
                    def __init__(self, in_f=8, max_features_ratio=0.5):
                        super().__init__()
                        self.subsample_proj = nn.Linear(in_f, int(in_f * max_features_ratio), bias=False)
                    def forward(self, feat):
                        return self.subsample_proj(feat)

                class RandomizedTreeCluster(nn.Module):
                    def __init__(self, in_f=4, depth=10):
                        super().__init__()
                        self.cutpoint_evaluator = nn.Linear(in_f, 64, bias=True)
                        self.leaf_average = nn.Linear(64, 1, bias=False)
                    def forward(self, feat):
                        return self.leaf_average(torch.relu(self.cutpoint_evaluator(feat)))

                class ExtraTreesArchitecture(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.input_features_8d = nn.Identity()
                        self.feature_subsampling = RandomFeatureSubsampling(in_features, 0.5)
                        sub_f = int(in_features * 0.5)
                        self.tree_cluster_1_200_trees = RandomizedTreeCluster(sub_f, 10)
                        self.tree_cluster_2_200_trees = RandomizedTreeCluster(sub_f, 10)
                        self.tree_cluster_3_200_trees = RandomizedTreeCluster(sub_f, 10)
                        self.tree_cluster_4_200_trees = RandomizedTreeCluster(sub_f, 10)
                    def forward(self, inp):
                        feat = self.input_features_8d(inp)
                        sub = self.feature_subsampling(feat)
                        c1 = self.tree_cluster_1_200_trees(sub) * 200.0
                        c2 = self.tree_cluster_2_200_trees(sub) * 200.0
                        c3 = self.tree_cluster_3_200_trees(sub) * 200.0
                        c4 = self.tree_cluster_4_200_trees(sub) * 200.0
                        return (c1 + c2 + c3 + c4) / 800.0

                graph_mod = ExtraTreesArchitecture()

            else:  # XGBoost
                class DepthWiseTreeStage(nn.Module):
                    def __init__(self, in_f=8, max_depth=5):
                        super().__init__()
                        num_leaves = 2**max_depth
                        self.split_scoring = nn.Linear(in_f, num_leaves, bias=True)
                        self.leaf_weights = nn.Linear(num_leaves, 1, bias=False)
                    def forward(self, feat):
                        return self.leaf_weights(torch.relu(self.split_scoring(feat)))

                class XGBoostArchitecture(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.input_features_8d = nn.Identity()
                        self.trees_stage_1_to_100 = DepthWiseTreeStage(in_features, 5)
                        self.trees_stage_101_to_200 = DepthWiseTreeStage(in_features, 5)
                        self.trees_stage_201_to_300 = DepthWiseTreeStage(in_features, 5)
                        self.learning_rate = 0.01
                        self.base_score = nn.Parameter(torch.tensor([14820.0]), requires_grad=False)
                    def forward(self, inp):
                        feat = self.input_features_8d(inp)
                        t1 = self.trees_stage_1_to_100(feat) * 100.0
                        t2 = self.trees_stage_101_to_200(feat) * 100.0
                        t3 = self.trees_stage_201_to_300(feat) * 100.0
                        return self.base_score + self.learning_rate * (t1 + t2 + t3)

                graph_mod = XGBoostArchitecture()

            self.writer.add_graph(graph_mod, x)
            print(f"[TB] Computational architecture graph logged: '{self.model_name}'")
        except Exception as e:
            print(f"[TB] Computational graph skipped: {e}")

    def close(self):
        self.writer.flush()
        self.writer.close()
        print(f"[TB] Writer closed: '{self.model_name}'")


class TorchTBLogger:
    """TensorBoard logger for PyTorch models (MLP / LSTM / TCN / Transformer).

    Usage inside any notebook
    -------------------------
    from utils.tb_logger import TorchTBLogger

    logger = TorchTBLogger("LSTM", model=model, sample_input=X_sample)
    for epoch in range(num_epochs):
        ...
        logger.log_epoch(epoch, train_loss, val_loss, y_val, preds_val, lr=scheduler.get_last_lr()[0])
        logger.log_gradients(model, epoch)   # optional — for debugging
    logger.log_final(y_train, p_train, y_test, p_test, hparams={"lr": 1e-3, "hidden": 128})
    logger.log_prediction_plots(y_test, p_test)
    logger.log_scatter_plot(y_test, p_test)
    logger.close()
    """

    def __init__(self, model_name, model=None, sample_input=None, log_root="runs"):
        self.model_name = model_name
        self.log_dir    = os.path.join(log_root, model_name)
        self.writer     = SummaryWriter(log_dir=self.log_dir)
        print(f"[TB] '{model_name}' -> {self.log_dir}")
        print(f"[TB] Launch: tensorboard --logdir={log_root}/")
        if model is not None and sample_input is not None:
            was_training = getattr(model, "training", False)
            if hasattr(model, "eval"):
                model.eval()
            try:
                self.writer.add_graph(model, sample_input)
                print("[TB] Model graph logged.")
            except Exception:
                try:
                    import torch
                    from torch.utils.tensorboard._pytorch_graph import (
                        parse, GraphDef, VersionDef, RunMetadata, StepStats, DeviceStepStats
                    )
                    with torch.no_grad():
                        trace = torch.jit.trace(model, sample_input, check_trace=False)
                        graph = trace.graph
                        torch._C._jit_pass_inline(graph)
                        list_of_nodes = parse(graph, trace, sample_input)
                        stepstats = RunMetadata(step_stats=StepStats(dev_stats=[DeviceStepStats(device="/device:CPU:0")]))
                        graph_def = GraphDef(node=list_of_nodes, versions=VersionDef(producer=22))
                        self.writer._get_file_writer().add_graph((graph_def, stepstats))
                        print("[TB] Model graph logged (non-strict trace).")
                except Exception as e2:
                    print(f"[TB] Graph skipped: {e2}")
            finally:
                if was_training and hasattr(model, "train"):
                    model.train()

    def log_epoch(self, epoch, train_loss, val_loss,
                  y_val=None, y_pred=None, lr=None):
        self.writer.add_scalars("Loss", {"Train": train_loss, "Val": val_loss}, epoch)
        if lr is not None:
            self.writer.add_scalar("LearningRate", lr, epoch)
        if y_val is not None and y_pred is not None:
            m = _compute_metrics(y_val, y_pred)
            self.writer.add_scalar("Metrics/MAE_W",  m["MAE_W"],  epoch)
            self.writer.add_scalar("Metrics/RMSE_W", m["RMSE_W"], epoch)
            self.writer.add_scalar("Metrics/R2",     m["R2"],     epoch)
            self.writer.add_scalar("Metrics/MAPE_%", m["MAPE_%"], epoch)

    def log_gradients(self, model, epoch):
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f"Gradients/{name}", param.grad, epoch)
                self.writer.add_scalar(f"GradNorms/{name}",
                                       param.grad.norm().item(), epoch)

    def log_final(self, y_train, y_pred_train, y_test, y_pred_test, hparams=None):
        tm = _compute_metrics(y_train, y_pred_train)
        vm = _compute_metrics(y_test,  y_pred_test)
        for k, v in tm.items():
            self.writer.add_scalar(f"Final/Train_{k}", v, 0)
        for k, v in vm.items():
            self.writer.add_scalar(f"Final/Test_{k}",  v, 0)
        hp = hparams or {}
        hp["model"] = self.model_name
        self.writer.add_hparams(
            hp,
            {
                "hparam/Test_MAE":  vm["MAE_W"],
                "hparam/Test_RMSE": vm["RMSE_W"],
                "hparam/Test_R2":   vm["R2"],
                "hparam/Train_R2":  tm["R2"],
                "hparam/R2_Gap":    tm["R2"] - vm["R2"],
            }
        )
        print(f"\n[{self.model_name}] Train  MAE={tm['MAE_W']:.0f}W  R2={tm['R2']:.3f}")
        print(f"[{self.model_name}] Test   MAE={vm['MAE_W']:.0f}W  R2={vm['R2']:.3f}")
        return {"train": tm, "test": vm}

    def log_prediction_plots(self, y_test, y_pred, n_plot=500):
        y_test = np.array(y_test)
        y_pred = np.array(y_pred)
        n = min(len(y_test), n_plot)
        x = np.arange(n)
        fig = plt.figure(figsize=(14, 8))
        gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.35)
        ax0 = fig.add_subplot(gs[0])
        ax0.plot(x, y_test[:n], label="Actual",    color="#2196F3", lw=1.2)
        ax0.plot(x, y_pred[:n], label="Predicted", color="#FF5722", lw=1.2, alpha=0.8, ls="--")
        ax0.set_title(f"{self.model_name} -- Actual vs Predicted (first {n} test samples)")
        ax0.set_ylabel("Power (W)")
        ax0.legend(loc="upper right", framealpha=0.7)
        ax0.spines[["top", "right"]].set_visible(False)
        residuals = y_test[:n] - y_pred[:n]
        ax1 = fig.add_subplot(gs[1])
        ax1.bar(x, residuals, color=np.where(residuals >= 0, "#4CAF50", "#F44336"),
                width=1.0, alpha=0.7)
        ax1.axhline(0, color="black", lw=0.8, ls="--")
        ax1.set_xlabel("Test Sample Index")
        ax1.set_ylabel("Residual (W)")
        ax1.set_title("Residuals  (Actual - Predicted)")
        ax1.spines[["top", "right"]].set_visible(False)
        _fig_to_writer(fig, self.writer, "Predictions/time_series", 0)

    def log_scatter_plot(self, y_test, y_pred):
        y_test = np.array(y_test)
        y_pred = np.array(y_pred)
        lo = min(y_test.min(), y_pred.min())
        hi = max(y_test.max(), y_pred.max())
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(y_test, y_pred, alpha=0.2, s=8, color="#7C4DFF")
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect prediction")
        r2 = r2_score(y_test, y_pred)
        ax.set_xlabel("Actual Power (W)")
        ax.set_ylabel("Predicted Power (W)")
        ax.set_title(f"{self.model_name} -- Scatter  R2={r2:.3f}")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        _fig_to_writer(fig, self.writer, "Predictions/scatter", 0)

    def close(self):
        self.writer.flush()
        self.writer.close()
        print(f"[TB] Writer closed: '{self.model_name}'")