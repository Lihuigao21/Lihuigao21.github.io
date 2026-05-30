#!/usr/bin/env python3
"""Run HamGNN with TensorBoard logging disabled.

This wrapper monkeypatches only the in-memory trainer setup for this process.
It does not modify the installed HamGNN source tree.
"""

from __future__ import annotations

import os

import pytorch_lightning as pl

try:
    from pytorch_lightning.loggers import LightningLoggerBase
except ImportError:  # PyTorch Lightning 1.x fallback
    from pytorch_lightning.loggers.base import LightningLoggerBase

import hamgnn.main as hm

_original_prepare_dataset = hm.prepare_dataset


class _NoOpExperiment:
    def add_text(self, *args, **kwargs):
        return None

    def add_hparams(self, *args, **kwargs):
        return None

    def add_scalar(self, *args, **kwargs):
        return None

    def add_figure(self, *args, **kwargs):
        return None


class _NoOpLogger(LightningLoggerBase):
    def __init__(self, save_dir):
        super().__init__()
        self._save_dir = save_dir
        self._log_dir = os.path.join(save_dir, self.name)

    @property
    def name(self):
        return "noop"

    @property
    def version(self):
        return ""

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def experiment(self):
        return _NoOpExperiment()

    def log_hyperparams(self, params, *args, **kwargs):
        return None

    def log_metrics(self, metrics, step=None):
        return None

    def save(self):
        return None

    def finalize(self, status):
        return None


def _setup_trainer_no_tensorboard(config, callbacks):
    callbacks = callbacks or []
    callbacks = [
        cb for cb in callbacks
        if cb.__class__.__name__ != "LearningRateMonitor"
    ]

    num_gpus = hm._normalize_num_gpus(getattr(config.setup, "num_gpus", None))
    requested_gpu_count = hm._count_requested_gpus(num_gpus)
    accelerator = getattr(config.setup, "accelerator", None)

    if not accelerator and requested_gpu_count > 1:
        accelerator = "ddp"

    logger = _NoOpLogger(config.profiler_params.train_dir)

    trainer_params = {
        "gpus": num_gpus,
        "precision": config.setup.precision,
        "callbacks": callbacks,
        "progress_bar_refresh_rate": 1,
        "logger": logger,
        "gradient_clip_val": config.optim_params.gradient_clip_val,
        "max_epochs": config.optim_params.max_epochs,
        "default_root_dir": config.profiler_params.train_dir,
        "min_epochs": config.optim_params.min_epochs,
    }

    if accelerator == "ddp":
        trainer_params["strategy"] = hm.DDPPlugin(static_graph=True)
    elif accelerator:
        trainer_params["strategy"] = accelerator

    if config.setup.resume and config.setup.checkpoint_path:
        trainer_params["resume_from_checkpoint"] = config.setup.checkpoint_path

    trainer = pl.Trainer(**trainer_params)
    return trainer, logger


def _prepare_dataset_split_test(config):
    if (
        os.environ.get("HAMGNN_TEST_USE_SPLIT") == "1"
        and config.setup.stage == "test"
    ):
        graph_data_path = config.dataset_params.graph_data_path
        if (
            not os.path.isfile(graph_data_path)
            and not graph_data_path.lower().endswith(".lmdb")
        ):
            graph_data_path = os.path.join(graph_data_path, "graph_data.npz")

        return hm.graph_data_module(
            dataset=graph_data_path,
            train_ratio=config.dataset_params.train_ratio,
            val_ratio=config.dataset_params.val_ratio,
            test_ratio=config.dataset_params.test_ratio,
            batch_size=config.dataset_params.batch_size,
            split_file=config.dataset_params.split_file,
            num_workers=getattr(config.dataset_params, "num_workers", 4),
            preload=getattr(config.dataset_params, "preload", 0),
            test_mode=False,
            data_format=getattr(config.dataset_params, "data_format", "auto"),
        )

    return _original_prepare_dataset(config)


hm.setup_trainer = _setup_trainer_no_tensorboard
hm.prepare_dataset = _prepare_dataset_split_test
hm.HamGNN()
