"""Small-N DD-FCN versus PINN-FCN implementation smoke for Zeng 2025."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


CASE_DIR = Path(__file__).resolve().parent
ROOT = CASE_DIR

from curve_quality import (
    curve_smoothness_gate,
    curve_smoothness_metrics,
)


def _json_default(value):
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError("Cannot JSON serialize {!r}".format(type(value)))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_config_hash(config):
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def discretize_ohmic(n_modes, omega_c, xi):
    """Paper Eqs. 23a-b with one-based mode indices."""

    index = np.arange(1, int(n_modes) + 1, dtype=np.float64)
    omega = float(omega_c) * np.log(
        float(n_modes) / (float(n_modes) - index + 0.5)
    )
    coupling = np.sqrt(float(xi) * float(omega_c) / float(n_modes)) * omega
    return omega, coupling


class SpinBosonCMM:
    """Primitive-MMST CMM dynamics for the paper's diabatic spin-boson model."""

    def __init__(self, model_config):
        self.n_modes = int(model_config["n_modes"])
        self.nstate = int(model_config["n_electronic_states"])
        self.epsilon = float(model_config["epsilon"])
        self.electronic_coupling = float(
            model_config["gamma_electronic_coupling"]
        )
        self.beta = float(model_config["beta"])
        self.mass = float(model_config["mass"])
        self.mapping_gamma = float(model_config["mapping_zpe_gamma"])
        self.project_mapping_after_rk4 = bool(
            model_config.get("project_mapping_after_rk4", False)
        )
        self.omega, self.coupling = discretize_ohmic(
            self.n_modes,
            model_config["omega_c"],
            model_config["kondo_xi"],
        )

    @property
    def state_size(self):
        return 2 * self.nstate + 2 * self.n_modes

    def sample_initial(self, ntraj, seed):
        rng = np.random.default_rng(int(seed))
        coth = 1.0 / np.tanh(0.5 * self.beta * self.omega)
        q_nuclear_std = np.sqrt(0.5 * coth / self.omega)
        p_nuclear_std = np.sqrt(0.5 * self.omega * coth)
        nuclear_q = rng.normal(size=(ntraj, self.n_modes)) * q_nuclear_std
        nuclear_p = rng.normal(size=(ntraj, self.n_modes)) * p_nuclear_std

        mapping = rng.normal(size=(ntraj, 2 * self.nstate))
        mapping /= np.linalg.norm(mapping, axis=1, keepdims=True)
        radius = math.sqrt(2.0 * (1.0 + self.nstate * self.mapping_gamma))
        mapping *= radius
        mapping_q = mapping[:, : self.nstate]
        mapping_p = mapping[:, self.nstate :]
        return np.concatenate((mapping_q, mapping_p, nuclear_q, nuclear_p), axis=1)

    def unpack(self, state):
        f = self.nstate
        n = self.n_modes
        return (
            state[..., :f],
            state[..., f : 2 * f],
            state[..., 2 * f : 2 * f + n],
            state[..., 2 * f + n : 2 * f + 2 * n],
        )

    def potential_and_gradient(self, nuclear_q):
        nuclear_q = np.asarray(nuclear_q, dtype=np.float64)
        leading = nuclear_q.shape[:-1]
        harmonic = 0.5 * np.sum((self.omega * nuclear_q) ** 2, axis=-1)
        bath_shift = np.sum(self.coupling * nuclear_q, axis=-1)
        potential = np.zeros(leading + (2, 2), dtype=np.float64)
        potential[..., 0, 0] = harmonic + self.epsilon - bath_shift
        potential[..., 1, 1] = harmonic - self.epsilon + bath_shift
        potential[..., 0, 1] = self.electronic_coupling
        potential[..., 1, 0] = self.electronic_coupling

        gradient = np.zeros(leading + (self.n_modes, 2, 2), dtype=np.float64)
        harmonic_gradient = self.omega**2 * nuclear_q
        gradient[..., :, 0, 0] = harmonic_gradient - self.coupling
        gradient[..., :, 1, 1] = harmonic_gradient + self.coupling
        return potential, gradient

    def derivative(self, state):
        mapping_q, mapping_p, nuclear_q, nuclear_p = self.unpack(state)
        potential, gradient = self.potential_and_gradient(nuclear_q)
        dq = np.einsum("...ij,...j->...i", potential, mapping_p)
        dp = -np.einsum("...ij,...j->...i", potential, mapping_q)
        dR = nuclear_p / self.mass
        action = 0.5 * (
            np.einsum("...i,...j->...ij", mapping_q, mapping_q)
            + np.einsum("...i,...j->...ij", mapping_p, mapping_p)
        )
        action[..., np.arange(self.nstate), np.arange(self.nstate)] -= (
            self.mapping_gamma
        )
        dP = -np.einsum("...nfg,...fg->...n", gradient, action)
        return np.concatenate((dq, dp, dR, dP), axis=-1)

    def rk4_step(self, state, dt):
        k1 = self.derivative(state)
        k2 = self.derivative(state + 0.5 * dt * k1)
        k3 = self.derivative(state + 0.5 * dt * k2)
        k4 = self.derivative(state + dt * k3)
        updated = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not self.project_mapping_after_rk4:
            return updated
        updated = np.array(updated, copy=True)
        mapping_q, mapping_p, _, _ = self.unpack(updated)
        current = 0.5 * np.sum(mapping_q**2 + mapping_p**2, axis=-1)
        target = 1.0 + self.nstate * self.mapping_gamma
        scale = np.sqrt(target / np.maximum(current, 1.0e-30))
        updated[..., : self.nstate] *= scale[..., None]
        updated[..., self.nstate : 2 * self.nstate] *= scale[..., None]
        return updated

    def propagate(self, initial, steps, dt):
        trajectory = np.empty(
            (initial.shape[0], int(steps) + 1, initial.shape[1]), dtype=np.float64
        )
        trajectory[:, 0] = initial
        state = initial.copy()
        for step in range(1, int(steps) + 1):
            state = self.rk4_step(state, float(dt))
            trajectory[:, step] = state
        return trajectory

    def mapping_sphere_value(self, state):
        mapping_q, mapping_p, _, _ = self.unpack(state)
        return 0.5 * np.sum(mapping_q**2 + mapping_p**2, axis=-1)

    def energy(self, state):
        mapping_q, mapping_p, nuclear_q, nuclear_p = self.unpack(state)
        potential, _ = self.potential_and_gradient(nuclear_q)
        action = 0.5 * (
            np.einsum("...i,...j->...ij", mapping_q, mapping_q)
            + np.einsum("...i,...j->...ij", mapping_p, mapping_p)
        )
        action[..., np.arange(self.nstate), np.arange(self.nstate)] -= (
            self.mapping_gamma
        )
        return 0.5 * np.sum(nuclear_p**2, axis=-1) / self.mass + np.einsum(
            "...ij,...ij->...", potential, action
        )

    def cmm_population(self, initial, trajectory, initial_state=0):
        q0, p0, _, _ = self.unpack(initial)
        qt, pt, _, _ = self.unpack(trajectory)
        initial_action = 0.5 * (q0**2 + p0**2) - self.mapping_gamma
        bar_gamma = (1.0 - self.mapping_gamma) / (
            1.0 + self.nstate * self.mapping_gamma
        )
        q_factor = (1.0 + self.nstate * bar_gamma) / (
            1.0 + self.nstate * self.mapping_gamma
        )
        final_action = q_factor * 0.5 * (qt**2 + pt**2) - bar_gamma
        weight = self.nstate * initial_action[:, int(initial_state)]
        return np.mean(weight[:, None, None] * final_action, axis=0)


class SigmoidMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, zero_output=False):
        super().__init__()
        layers = []
        previous = int(input_size)
        for width in hidden_sizes:
            layers.extend((nn.Linear(previous, int(width)), nn.Sigmoid()))
            previous = int(width)
        final = nn.Linear(previous, int(output_size))
        if zero_output:
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        layers.append(final)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DDResidualFCN(nn.Module):
    def __init__(self, state_mean, state_std, delta_scale, hidden_sizes):
        super().__init__()
        state_size = int(state_mean.numel())
        self.register_buffer("state_mean", state_mean.clone())
        self.register_buffer("state_std", state_std.clone())
        self.register_buffer("delta_scale", delta_scale.clone())
        self.net = SigmoidMLP(state_size, state_size, hidden_sizes)

    def forward(self, state):
        normalized = (state - self.state_mean) / self.state_std
        return state + self.delta_scale * self.net(normalized)


class PrimitiveMMSTPINN(nn.Module):
    def __init__(
        self,
        n_modes,
        nstate,
        dt,
        nuclear_mean,
        nuclear_std,
        hidden_sizes,
        project_mapping_after_step=False,
    ):
        super().__init__()
        self.n_modes = int(n_modes)
        self.nstate = int(nstate)
        self.dt = float(dt)
        self.project_mapping_after_step = bool(project_mapping_after_step)
        self.register_buffer("nuclear_mean", nuclear_mean.clone())
        self.register_buffer("nuclear_std", nuclear_std.clone())
        self.potential_net = SigmoidMLP(
            self.n_modes,
            self.nstate * self.nstate,
            hidden_sizes,
            zero_output=True,
        )
        self.gradient_net = SigmoidMLP(
            self.n_modes,
            self.n_modes * self.nstate * self.nstate,
            hidden_sizes,
            zero_output=True,
        )
        self.log_mass = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.mapping_gamma = nn.Parameter(torch.tensor(0.4, dtype=torch.float32))

    @property
    def mass(self):
        return torch.exp(self.log_mass)

    def unpack(self, state):
        f = self.nstate
        n = self.n_modes
        return (
            state[..., :f],
            state[..., f : 2 * f],
            state[..., 2 * f : 2 * f + n],
            state[..., 2 * f + n : 2 * f + 2 * n],
        )

    def learned_fields(self, nuclear_q):
        normalized = (nuclear_q - self.nuclear_mean) / self.nuclear_std
        potential = self.potential_net(normalized).reshape(-1, self.nstate, self.nstate)
        potential = 0.5 * (potential + potential.transpose(-1, -2))
        gradient = self.gradient_net(normalized).reshape(
            -1, self.n_modes, self.nstate, self.nstate
        )
        gradient = 0.5 * (gradient + gradient.transpose(-1, -2))
        return potential, gradient

    def derivative(self, state):
        mapping_q, mapping_p, nuclear_q, nuclear_p = self.unpack(state)
        potential, gradient = self.learned_fields(nuclear_q)
        dq = torch.einsum("bij,bj->bi", potential, mapping_p)
        dp = -torch.einsum("bij,bj->bi", potential, mapping_q)
        dR = nuclear_p / self.mass
        action = 0.5 * (
            torch.einsum("bi,bj->bij", mapping_q, mapping_q)
            + torch.einsum("bi,bj->bij", mapping_p, mapping_p)
        )
        identity = torch.eye(self.nstate, device=state.device, dtype=state.dtype)
        action = action - self.mapping_gamma * identity[None]
        dP = -torch.einsum("bnfg,bfg->bn", gradient, action)
        return torch.cat((dq, dp, dR, dP), dim=-1)

    def forward(self, state):
        dt = self.dt
        k1 = self.derivative(state)
        k2 = self.derivative(state + 0.5 * dt * k1)
        k3 = self.derivative(state + 0.5 * dt * k2)
        k4 = self.derivative(state + dt * k3)
        updated = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not self.project_mapping_after_step:
            return updated
        mapping_q, mapping_p, nuclear_q, nuclear_p = self.unpack(updated)
        current = 0.5 * torch.sum(mapping_q**2 + mapping_p**2, dim=-1)
        target = torch.clamp(1.0 + self.nstate * self.mapping_gamma, min=1.0e-6)
        scale = torch.sqrt(target / torch.clamp(current, min=1.0e-30))
        return torch.cat(
            (
                mapping_q * scale[:, None],
                mapping_p * scale[:, None],
                nuclear_q,
                nuclear_p,
            ),
            dim=-1,
        )


def _trajectory_pairs(trajectory):
    return (
        trajectory[:, :-1].reshape(-1, trajectory.shape[-1]),
        trajectory[:, 1:].reshape(-1, trajectory.shape[-1]),
    )


def _save_dataset(path, payload, config_hash):
    np.savez_compressed(path, config_hash=np.asarray(config_hash), **payload)


def _load_or_generate_dataset(path, config, config_hash):
    if path.exists():
        with np.load(path, allow_pickle=False) as existing:
            saved_hash = str(existing["config_hash"].item())
            if saved_hash != config_hash:
                raise RuntimeError("Existing dataset config hash does not match.")
            required = ("train", "validation", "test_0", "test_1")
            if all(key in existing and np.all(np.isfinite(existing[key])) for key in required):
                return {key: np.array(existing[key], copy=True) for key in required}, True

    model = SpinBosonCMM(config["model"])
    data = config["data"]
    seed = int(data["seed"])
    steps = int(data["trajectory_steps"])
    dt = float(data["dt"])
    counts = {
        "train": int(data["train_trajectories"]),
        "validation": int(data["validation_trajectories"]),
        "test_0": int(data["test_trajectories_per_batch"]),
        "test_1": int(data["test_trajectories_per_batch"]),
    }
    payload = {}
    for offset, (name, count) in enumerate(counts.items()):
        initial = model.sample_initial(count, seed + 1000 * offset)
        payload[name] = model.propagate(initial, steps, dt)
    _save_dataset(path, payload, config_hash)
    return payload, False


def _make_scalers(train):
    x, y = _trajectory_pairs(train)
    state_mean = np.mean(x, axis=0)
    state_std = np.maximum(np.std(x, axis=0), 1.0e-6)
    delta_scale = np.maximum(np.std(y - x, axis=0), 1.0e-6)
    return state_mean, state_std, delta_scale


def _normalized_increment_loss(prediction, target, current, delta_scale):
    return torch.mean(((prediction - current - (target - current)) / delta_scale) ** 2)


def _train_model(
    name,
    model,
    train,
    validation,
    delta_scale,
    config,
    config_hash,
    output,
    device,
):
    train_config = config["training"]
    epochs = int(train_config["epochs"])
    batch_size = int(train_config["batch_size"])
    x_train_np, y_train_np = _trajectory_pairs(train)
    x_val_np, y_val_np = _trajectory_pairs(validation)
    x_train = torch.from_numpy(x_train_np.astype(np.float32))
    y_train = torch.from_numpy(y_train_np.astype(np.float32))
    x_val = torch.from_numpy(x_val_np.astype(np.float32)).to(device)
    y_val = torch.from_numpy(y_val_np.astype(np.float32)).to(device)
    delta_scale_t = torch.from_numpy(delta_scale.astype(np.float32)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    checkpoint = output / (name + "_latest.pt")
    best_checkpoint = output / (name + "_best.pt")
    history = []
    start_epoch = 0
    best_val = float("inf")
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device)
        if saved.get("config_hash") != config_hash:
            raise RuntimeError("{} checkpoint config hash mismatch".format(name))
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        history = list(saved["history"])
        start_epoch = int(saved["epoch"]) + 1
        best_val = float(saved["best_val"])

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(train_config["torch_seed"]) + (0 if name == "dd" else 1))
    model.to(device)
    started = time.perf_counter()
    for epoch in range(start_epoch, epochs):
        model.train()
        permutation = torch.randperm(x_train.shape[0], generator=generator)
        train_loss_sum = 0.0
        samples = 0
        for first in range(0, x_train.shape[0], batch_size):
            index = permutation[first : first + batch_size]
            xb = x_train[index].to(device)
            yb = y_train[index].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb)
            loss = _normalized_increment_loss(prediction, yb, xb, delta_scale_t)
            if isinstance(model, PrimitiveMMSTPINN):
                f = model.nstate
                sphere = 0.5 * torch.sum(
                    xb[:, :f] ** 2 + xb[:, f : 2 * f] ** 2, dim=-1
                )
                target_sphere = 1.0 + f * model.mapping_gamma
                loss = loss + float(
                    train_config.get("mapping_constraint_weight", 0.0)
                ) * torch.mean((sphere - target_sphere) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(train_config["gradient_clip_norm"])
            )
            optimizer.step()
            count = int(xb.shape[0])
            train_loss_sum += float(loss.detach()) * count
            samples += count

        model.eval()
        with torch.no_grad():
            prediction = model(x_val)
            val_loss = float(
                _normalized_increment_loss(prediction, y_val, x_val, delta_scale_t)
            )
        train_loss = train_loss_sum / samples
        record = {
            "epoch": epoch,
            "train_increment_nmse": train_loss,
            "validation_increment_nmse": val_loss,
        }
        if isinstance(model, PrimitiveMMSTPINN):
            record["learned_mass"] = float(model.mass.detach())
            record["learned_mapping_gamma"] = float(model.mapping_gamma.detach())
        history.append(record)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "config_hash": config_hash,
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "best_val": best_val,
                },
                best_checkpoint,
            )
        torch.save(
            {
                "config_hash": config_hash,
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "best_val": best_val,
            },
            checkpoint,
        )
        if epoch == start_epoch or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(
                "{} epoch {}/{} train={:.6g} val={:.6g}".format(
                    name, epoch + 1, epochs, train_loss, val_loss
                ),
                flush=True,
            )

    best = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(best["model"])
    return {
        "history": history,
        "best_epoch": int(best["epoch"]),
        "best_validation_increment_nmse": float(best["best_val"]),
        "wall_seconds": time.perf_counter() - started,
        "latest_checkpoint": checkpoint,
        "best_checkpoint": best_checkpoint,
    }


def _rollout(model, initial, steps, device):
    state = torch.from_numpy(initial.astype(np.float32)).to(device)
    values = [state.detach().cpu().numpy().astype(np.float64)]
    model.eval()
    with torch.no_grad():
        for _ in range(int(steps)):
            state = model(state)
            values.append(state.detach().cpu().numpy().astype(np.float64))
    return np.stack(values, axis=1)


def _state_nrmse(prediction, reference):
    scale = np.maximum(np.std(reference.reshape(-1, reference.shape[-1]), axis=0), 1.0e-8)
    squared = np.mean(((prediction - reference) / scale) ** 2, axis=(0, 2))
    return np.sqrt(squared)


def _population_metrics(reference, prediction):
    difference = prediction - reference
    return {
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "max_abs_difference": float(np.max(np.abs(difference))),
    }


def _curve_audit(times, curves, thresholds):
    records = {}
    all_pass = True
    for name, values in curves.items():
        metrics = curve_smoothness_metrics(times, values)
        passed = curve_smoothness_gate(
            metrics,
            max_rms_fraction=thresholds["curve_roughness_rms_fraction"],
            max_peak_fraction=thresholds["curve_roughness_max_fraction"],
            max_spike_fraction=thresholds["curve_isolated_spike_fraction"],
            max_endpoint_roughness_fraction=thresholds[
                "curve_endpoint_roughness_max_fraction"
            ],
        )
        records[name] = {"metrics": metrics, "pass": passed}
        all_pass = all_pass and passed
    return {"series": records, "all_pass": bool(all_pass)}


def _write_history(path, histories):
    fields = [
        "method",
        "epoch",
        "train_increment_nmse",
        "validation_increment_nmse",
        "learned_mass",
        "learned_mapping_gamma",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, result in histories.items():
            for record in result["history"]:
                writer.writerow({"method": method, **record})


def _write_population_csv(path, times, populations):
    fields = ["time"]
    for name in populations:
        fields.extend((name + "_population_0", name + "_population_1"))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, time_value in enumerate(times):
            row = {"time": float(time_value)}
            for name, values in populations.items():
                row[name + "_population_0"] = float(values[index, 0])
                row[name + "_population_1"] = float(values[index, 1])
            writer.writerow(row)


def _write_plot(path, times, populations, nrmse):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    colors = {"reference": "black", "dd": "#e68a00", "pinn": "#d62728"}
    labels = {"reference": "Reference CMM", "dd": "DD-FCN", "pinn": "PINN-FCN"}
    for name in ("reference", "dd", "pinn"):
        axes[0].plot(
            times,
            populations[name][:, 0],
            color=colors[name],
            label=labels[name],
        )
    axes[1].plot(times, nrmse["dd"], color=colors["dd"], label=labels["dd"])
    axes[1].plot(times, nrmse["pinn"], color=colors["pinn"], label=labels["pinn"])
    axes[0].set(xlabel="time (a.u.)", ylabel="CMM population, state 1")
    axes[1].set(xlabel="time (a.u.)", ylabel="full-state normalized RMSE")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle("Zeng 2025 small-N DD/PINN implementation smoke")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _environment_record():
    package_names = (
        "torch",
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "tensorboard",
        "tqdm",
    )
    packages = {}
    for name in package_names:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_build": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "packages": packages,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_smoke.json")
    parser.add_argument("--output", default="scratch/smoke_v1")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = CASE_DIR / config_path
    output = Path(args.output)
    if not output.is_absolute():
        output = CASE_DIR / output
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_hash = _canonical_config_hash(config)
    thresholds = config["predeclared_thresholds"]
    started = time.perf_counter()
    status_path = output / "status.json"
    status_path.write_text(
        json.dumps(
            {
                "status": "running",
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "config_hash": config_hash,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if config["training"]["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device(config["training"]["device"])
    torch.manual_seed(int(config["training"]["torch_seed"]))
    np.random.seed(int(config["data"]["seed"]))

    dataset_path = output / "dataset.npz"
    dataset, reused_dataset = _load_or_generate_dataset(
        dataset_path, config, config_hash
    )
    dynamics = SpinBosonCMM(config["model"])
    state_mean, state_std, delta_scale = _make_scalers(dataset["train"])
    hidden = config["network"]["hidden_sizes"]
    dd = DDResidualFCN(
        torch.from_numpy(state_mean.astype(np.float32)),
        torch.from_numpy(state_std.astype(np.float32)),
        torch.from_numpy(delta_scale.astype(np.float32)),
        hidden,
    )
    nuclear_slice = slice(2 * dynamics.nstate, 2 * dynamics.nstate + dynamics.n_modes)
    pinn = PrimitiveMMSTPINN(
        dynamics.n_modes,
        dynamics.nstate,
        config["data"]["dt"],
        torch.from_numpy(state_mean[nuclear_slice].astype(np.float32)),
        torch.from_numpy(state_std[nuclear_slice].astype(np.float32)),
        hidden,
        project_mapping_after_step=bool(
            config["network"].get("pinn_project_mapping_after_step", False)
        ),
    )
    histories = {
        "dd": _train_model(
            "dd",
            dd,
            dataset["train"],
            dataset["validation"],
            delta_scale,
            config,
            config_hash,
            output,
            device,
        ),
        "pinn": _train_model(
            "pinn",
            pinn,
            dataset["train"],
            dataset["validation"],
            delta_scale,
            config,
            config_hash,
            output,
            device,
        ),
    }

    steps = int(config["data"]["trajectory_steps"])
    dt = float(config["data"]["dt"])
    times = np.arange(steps + 1, dtype=np.float64) * dt
    reference_batches = []
    dd_batches = []
    pinn_batches = []
    for batch in range(int(config["data"]["test_batches"])):
        reference = dataset["test_{}".format(batch)]
        initial = reference[:, 0]
        reference_batches.append(reference)
        dd_batches.append(_rollout(dd.to(device), initial, steps, device))
        pinn_batches.append(_rollout(pinn.to(device), initial, steps, device))
    reference_all = np.concatenate(reference_batches, axis=0)
    dd_all = np.concatenate(dd_batches, axis=0)
    pinn_all = np.concatenate(pinn_batches, axis=0)

    populations = {
        "reference": dynamics.cmm_population(reference_all[:, 0], reference_all),
        "dd": dynamics.cmm_population(reference_all[:, 0], dd_all),
        "pinn": dynamics.cmm_population(reference_all[:, 0], pinn_all),
    }
    nrmse = {
        "dd": _state_nrmse(dd_all, reference_all),
        "pinn": _state_nrmse(pinn_all, reference_all),
    }
    population_metrics = {
        "dd": _population_metrics(populations["reference"], populations["dd"]),
        "pinn": _population_metrics(populations["reference"], populations["pinn"]),
    }

    fine_initial = reference_batches[0][:, 0]
    fine_reference = dynamics.propagate(
        fine_initial, 2 * steps, float(config["data"]["fine_dt"])
    )
    fine_population = dynamics.cmm_population(fine_initial, fine_reference[:, ::2])
    dt_population_difference = float(
        np.max(np.abs(fine_population - dynamics.cmm_population(fine_initial, reference_batches[0])))
    )

    target_sphere = 1.0 + dynamics.nstate * dynamics.mapping_gamma
    initial_sphere_error = float(
        np.max(np.abs(dynamics.mapping_sphere_value(dataset["train"][:, 0]) - target_sphere))
    )
    reference_sphere_drift = float(
        np.max(
            np.abs(
                dynamics.mapping_sphere_value(reference_all)
                - dynamics.mapping_sphere_value(reference_all[:, :1])
            )
        )
    )
    pinn_sphere_drift = float(
        np.max(
            np.abs(
                dynamics.mapping_sphere_value(pinn_all)
                - dynamics.mapping_sphere_value(pinn_all[:, :1])
            )
        )
    )
    reference_energy = dynamics.energy(reference_all)
    reference_energy_scale = np.maximum(np.abs(reference_energy[:, :1]), 1.0e-12)
    reference_energy_drift = float(
        np.max(np.abs(reference_energy - reference_energy[:, :1]) / reference_energy_scale)
    )
    test_batch_difference = {}
    for name, batches in (
        ("reference", reference_batches),
        ("dd", dd_batches),
        ("pinn", pinn_batches),
    ):
        first = dynamics.cmm_population(batches[0][:, 0], batches[0])
        second = dynamics.cmm_population(reference_batches[1][:, 0], batches[1])
        test_batch_difference[name] = float(np.max(np.abs(first - second)))

    curve_quality = _curve_audit(
        times,
        {
            "reference_population_0": populations["reference"][:, 0],
            "dd_population_0": populations["dd"][:, 0],
            "pinn_population_0": populations["pinn"][:, 0],
            "dd_state_nrmse": nrmse["dd"],
            "pinn_state_nrmse": nrmse["pinn"],
        },
        thresholds,
    )
    finite = bool(
        all(np.all(np.isfinite(values)) for values in populations.values())
        and all(np.all(np.isfinite(values)) for values in nrmse.values())
    )
    gates = {
        "finite_all_outputs": finite,
        "initial_mapping_sphere": initial_sphere_error
        <= thresholds["initial_mapping_sphere_max_abs_error"],
        "reference_mapping_sphere": reference_sphere_drift
        <= thresholds["reference_mapping_sphere_max_abs_drift"],
        "reference_energy": reference_energy_drift
        <= thresholds["reference_relative_energy_max_abs_drift"],
        "reference_dt_halving": dt_population_difference
        <= thresholds["reference_dt_halving_population_max_abs_difference"],
        "pinn_validation": math.sqrt(
            histories["pinn"]["best_validation_increment_nmse"]
        )
        <= thresholds["pinn_validation_increment_nrmse"],
        "pinn_rollout": float(np.max(nrmse["pinn"]))
        <= thresholds["pinn_rollout_state_nrmse"],
        "pinn_population": population_metrics["pinn"]["rmse"]
        <= thresholds["pinn_population_rmse"],
        "pinn_mapping_sphere": pinn_sphere_drift
        <= thresholds["pinn_mapping_sphere_max_abs_drift"],
        "pinn_not_worse_than_dd": population_metrics["pinn"]["rmse"]
        <= thresholds["pinn_population_rmse_not_worse_than_dd_factor"]
        * population_metrics["dd"]["rmse"],
        "independent_test_batches": max(test_batch_difference.values())
        <= thresholds["independent_test_batch_population_max_abs_difference"],
        "curve_quality": curve_quality["all_pass"],
    }

    history_path = output / "training_history.csv"
    population_path = output / "population_curves.csv"
    plot_path = output / "diagnostic.png"
    _write_history(history_path, histories)
    _write_population_csv(population_path, times, populations)
    _write_plot(plot_path, times, populations, nrmse)
    source_paths = [Path(__file__).resolve(), config_path]
    report = {
        "status": "verified_smoke" if all(gates.values()) else "partial_smoke_failed_gates",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - started,
        "config_hash": config_hash,
        "config": config,
        "dataset_reused": reused_dataset,
        "environment": _environment_record(),
        "model_discretization": {
            "omega": dynamics.omega,
            "coupling": dynamics.coupling,
        },
        "training": {
            name: {
                key: value
                for key, value in result.items()
                if key != "history"
            }
            for name, result in histories.items()
        },
        "learned_pinn_parameters": {
            "mass": float(pinn.mass.detach().cpu()),
            "mapping_zpe_gamma": float(pinn.mapping_gamma.detach().cpu()),
        },
        "metrics": {
            "initial_mapping_sphere_max_abs_error": initial_sphere_error,
            "reference_mapping_sphere_max_abs_drift": reference_sphere_drift,
            "pinn_mapping_sphere_max_abs_drift": pinn_sphere_drift,
            "reference_relative_energy_max_abs_drift": reference_energy_drift,
            "reference_dt_halving_population_max_abs_difference": dt_population_difference,
            "test_batch_population_max_abs_difference": test_batch_difference,
            "population": population_metrics,
            "dd_rollout_state_nrmse_max": float(np.max(nrmse["dd"])),
            "pinn_rollout_state_nrmse_max": float(np.max(nrmse["pinn"])),
            "curve_quality": curve_quality,
        },
        "gates": gates,
        "all_gates_pass": bool(all(gates.values())),
        "artifacts": {
            "dataset": {"path": dataset_path, "sha256": _sha256(dataset_path)},
            "training_history": {"path": history_path, "sha256": _sha256(history_path)},
            "population_curves": {"path": population_path, "sha256": _sha256(population_path)},
            "diagnostic_plot": {"path": plot_path, "sha256": _sha256(plot_path)},
        },
        "provenance": {
            "git_commit": _git_commit(),
            "source_sha256": {str(path): _sha256(path) for path in source_paths},
        },
    }
    audit_path = output / "audit.json"
    audit_path.write_text(
        json.dumps(report, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    status_path.write_text(
        json.dumps(
            {
                "status": report["status"],
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "config_hash": config_hash,
                "all_gates_pass": report["all_gates_pass"],
                "audit": str(audit_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "all_gates_pass": report["all_gates_pass"],
                "wall_seconds": report["wall_seconds"],
                "metrics": report["metrics"],
                "gates": gates,
                "audit": str(audit_path),
            },
            indent=2,
            default=_json_default,
        )
    )


if __name__ == "__main__":
    main()
