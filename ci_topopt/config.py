"""Configuration and loading path."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np


DEFAULTS = {
    "geometry": {
        "length": 10.0, "nx": 100, "ny": 100, "corner_size": 0.4,
        "port_half_width": 1.0, "subport_width": 0.35,
        "passive_frame": True, "frame_width": 0.2,
    },
    "material": {"young": 40.0, "poisson": 0.45, "penalty": 3.0, "epsilon": 1e-6},
    "huhu": {"alpha": 2e-6},
    "newton": {
        "rtol": 1e-7, "atol": 1e-9, "max_it": 80,
        "min_step": 1e-5, "max_step": 0.025, "min_j": 1e-8,
    },
    "loading": {"stroke": 1.0, "fractions": [0.25, 0.5, 0.75, 1.0], "gain": 1.0},
    "optimization": {
        "max_iter": 400, "volume_fraction": 0.5, "initial_density": 0.5,
        "energy_bound": 1.5, "filter_radius": 0.25, "move": 0.03,
        "beta_init": 1.0, "beta_max": 64.0, "beta_interval": 50,
        "design_tolerance": 1e-4, "constraint_tolerance": 1e-4,
        "objective_tolerance": 1e-6,
    },
    "output": {"directory": "results/inverter", "save_every": 10},
}


def load_config(path=None):
    config = deepcopy(DEFAULTS)
    if path is not None:
        incoming = json.loads(Path(path).read_text())
        if not isinstance(incoming, dict):
            raise ValueError("Configuration must be a JSON object.")
        for section, values in incoming.items():
            if section not in config or not isinstance(values, dict):
                raise ValueError(f"Unknown or invalid configuration section: {section}")
            unknown = set(values) - set(config[section]) - ({"weights"} if section == "loading" else set())
            if unknown:
                raise ValueError(f"Unknown {section} settings: {sorted(unknown)}")
            config[section].update(values)
    validate_config(config)
    return config


def validate_config(config):
    for section, values in config.items():
        for key, value in values.items():
            if isinstance(value, (int, float)) and not np.isfinite(value):
                raise ValueError(f"{section}.{key} must be finite.")
    geo, mat, opt = (config[key] for key in ("geometry", "material", "optimization"))
    for key in ("nx", "ny"):
        if not isinstance(geo[key], int) or geo[key] < 4:
            raise ValueError(f"geometry.{key} must be an integer >= 4.")
    length = geo["length"]
    if length <= 0 or not 0 < geo["corner_size"] < length / 2:
        raise ValueError("Invalid domain length or corner support size.")
    if not 0 < geo["subport_width"] <= geo["port_half_width"] < length / 2:
        raise ValueError("Invalid input/output port dimensions.")
    if geo["corner_size"] >= length / 2 - geo["port_half_width"]:
        raise ValueError("Corner supports must not overlap the input port.")
    if geo["frame_width"] <= 0:
        raise ValueError("geometry.frame_width must be positive.")
    if mat["young"] <= 0 or not -1 < mat["poisson"] < 0.5:
        raise ValueError("Invalid elastic material parameters.")
    if mat["penalty"] < 1 or not 0 < mat["epsilon"] < 1:
        raise ValueError("Invalid SIMP parameters.")
    if config["huhu"]["alpha"] <= 0:
        raise ValueError("huhu.alpha must be positive.")
    for key in ("volume_fraction", "initial_density"):
        if not 0 < opt[key] < 1:
            raise ValueError(f"optimization.{key} must lie in (0, 1).")
    for key in ("max_iter", "beta_interval"):
        if not isinstance(opt[key], int) or opt[key] <= 0:
            raise ValueError(f"optimization.{key} must be a positive integer.")
    for key in ("energy_bound", "filter_radius", "move", "beta_init", "design_tolerance",
                "constraint_tolerance", "objective_tolerance"):
        if opt[key] <= 0:
            raise ValueError(f"optimization.{key} must be positive.")
    if opt["move"] > 1 or opt["beta_max"] < opt["beta_init"]:
        raise ValueError("Invalid move limit or projection continuation.")
    newton = config["newton"]
    if not isinstance(newton["max_it"], int) or newton["max_it"] <= 0:
        raise ValueError("newton.max_it must be a positive integer.")
    if any(newton[key] <= 0 for key in ("rtol", "atol", "min_j", "min_step", "max_step")):
        raise ValueError("Newton tolerances and increments must be positive.")
    if newton["min_step"] > newton["max_step"] or newton["min_j"] >= 1:
        raise ValueError("Invalid Newton increment or Jacobian threshold.")
    if not isinstance(config["output"]["save_every"], int) or config["output"]["save_every"] <= 0:
        raise ValueError("output.save_every must be a positive integer.")
    if not isinstance(config["output"]["directory"], str) or not config["output"]["directory"].strip():
        raise ValueError("output.directory must be nonempty.")
    loading_path(config)


def loading_path(config):
    loading = config["loading"]
    stroke = float(loading["stroke"])
    fractions = np.asarray(loading["fractions"], dtype=float)
    if not np.isfinite(stroke) or stroke == 0:
        raise ValueError("loading.stroke must be finite and nonzero.")
    if (fractions.ndim != 1 or fractions.size < 2 or not np.all(np.isfinite(fractions))
            or fractions[0] <= 0 or np.any(np.diff(fractions) <= 0)
            or fractions[-1] != 1.0):
        raise ValueError("Use at least two increasing fractions in (0, 1], ending at 1.")
    weights = np.asarray(loading.get("weights", np.ones(fractions.size)), dtype=float)
    if weights.shape != fractions.shape or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Loading weights must be finite, positive, and match the fractions.")
    if not np.isfinite(loading["gain"]) or loading["gain"] <= 0:
        raise ValueError("loading.gain must be positive.")
    return stroke * fractions, weights / weights.sum()
