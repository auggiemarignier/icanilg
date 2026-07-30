"""Solve the IC anisotropy problem modelled as a purely linear gaussian system."""

import argparse
import datetime
import logging
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from linear_gaussian import (
    calc_log_evidence,
    calc_posterior_cov,
    calc_posterior_mean,
)
from raytracer import SphericalMesh
from tti.elastic.voigt import (
    gradient_C_wrt_A,
    gradient_C_wrt_C,
    gradient_C_wrt_F,
    gradient_C_wrt_L,
    gradient_C_wrt_N,
)
from tti.traveltimes.traveltimes import (
    calculate_path_direction_vector,
    calculate_relative_traveltime_voigt,
)

from config import load_config, save_resolved_config
from config.builders import make_builder
from config.components import register_builder
from config.models import config_to_json_dict

# basic module logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def lonlatrad_to_xyz(lonlatrad: np.ndarray) -> np.ndarray:
    """Convert (lon, lat, radius) to Cartesian (x, y, z) coordinates.

    Parameters
    ----------
    lonlatrad : np.ndarray, shape (..., 3)
        Array of longitude (degrees), latitude (degrees), and radius (km).

    Returns
    -------
    xyz : np.ndarray, shape (..., 3)
        Array of Cartesian coordinates in km.
    """
    lon = np.radians(lonlatrad[..., 0])
    lat = np.radians(lonlatrad[..., 1])
    r = lonlatrad[..., 2]

    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)

    return np.stack([x, y, z], axis=-1)


def determine_weights(
    mesh: SphericalMesh, ic_in: np.ndarray, path_directions: np.ndarray
) -> np.ndarray:
    """Determine weights for each path based on the distance travelled in each region.

    Parameters
    ----------
    mesh : SphericalMesh
        The mesh defining the geometry and properties of the inner core.
    ic_in : ndarray, shape (num_paths, 3)
        Entry points of paths into the inner core (longitude (deg), latitude (deg), radius (km)).
    path_directions : ndarray, shape (num_paths, 3)
        Direction vectors for each path.

    Returns
    -------
    weights : ndarray, shape (1, num_segments, num_paths)
        Fractional distance of each path in each segment.  Additional axis for broadcasting with travel time calculator.
    """
    segment_distances = mesh.ray_distances_per_region(
        lonlatrad_to_xyz(ic_in), path_directions
    )
    total_distances = segment_distances.sum(axis=1)
    weights = segment_distances / total_distances[:, None]
    ws = weights.T[None, ...]
    logger.debug(
        "determine_weights: segment_distances.shape=%s total_distances.shape=%s",
        segment_distances.shape,
        total_distances.shape,
    )
    logger.debug(
        "determine_weights: computed weights for %d paths and %d segments",
        ic_in.shape[0],
        segment_distances.shape[1],
    )
    return ws


def construct_forward_map(
    path_directions: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Constructs the linear map from Love parameters to travel time.

    The input to the forward mapping is a stack of Love parameters, 3 (A,C,F) for each cell in the mesh.

    Combines various bits a pieces:
    1) Add 0 shear components (L, N) to input vector
    2) Mapping from a vector of elastic parameters to Voigt elastic tensor
        Making use of the derivative objects in `tti.elastic.voigt` a basis can be constructed
            basis = np.stack([dCdA, dCdC, dCdF, dCdL, dCdN], axis=0)
    3) With the `path_directions` compute the travel time in each cell as in `tti.traveltimes.traveltimes.calculate_relative_traveltime_voigt`
    4) With the `weights` calculated from `determine_weights` perform a weighted sum along each path as in `tti.traveltimes.traveltimes.TravelTimeCalculator._call_core.

    These four steps are to be combined into a single matrix that this function returns.
    """

    """Analytic forward matrix from 3 Love params/segment -> fractional travel-time (no rotations)."""

    # Map from the 5 dC basis (A,C,F,L,N) to the 3 model params (A,C,F).
    T = np.array(
        [
            [1.0, 0.0, 0.0],  # dC/dA
            [0.0, 1.0, 0.0],  # dC/dC
            [0.0, 0.0, 1.0],  # dC/dF
            [0.0, 0.0, 0.0],  # dC/dL (not modelled)
            [0.0, 0.0, 0.0],  # dC/dN (not modelled)
        ],
        dtype=float,
    )  # shape (5,3)

    n_segments = weights.shape[1]
    n_paths = weights.shape[2]
    n_params_per_seg = 3  # A,C,F

    # 5 constant dC basis tensors (5,6,6)
    dC_stack = np.stack(
        [
            gradient_C_wrt_A(),
            gradient_C_wrt_C(),
            gradient_C_wrt_F(),
            gradient_C_wrt_L(),
            gradient_C_wrt_N(),
        ],
        axis=0,
    )

    # Broadcast dC_stack to (n_segments,5,6,6)
    dC_b = np.broadcast_to(dC_stack[None, ...], (n_segments, *dC_stack.shape))

    # Contract with ray-direction outer-products -> (n_segments, 5, n_paths)
    dt_dC = calculate_relative_traveltime_voigt(
        path_directions, dC_b, normalisation=0.5
    )

    logger.debug(
        "construct_forward_map: weights.shape=%s path_directions.shape=%s",
        weights.shape,
        path_directions.shape,
    )
    logger.debug(
        "construct_forward_map: dC_stack.shape=%s dt_dC.shape=%s",
        dC_stack.shape,
        dt_dC.shape,
    )

    # Map from 5 dC-basis -> 3 model params: tensordot over basis axis
    # result shape -> (n_segments, n_paths, 3) -> transpose to (n_segments, 3, n_paths)
    dt_model = np.tensordot(dt_dC, T, axes=([1], [0])).transpose(0, 2, 1)

    # Apply fractional path-length weights per segment/path and flatten to (n_paths, n_segments*3)
    seg_weights = weights[0, :, :]  # (n_segments, n_paths)
    weighted = seg_weights[:, None, :] * dt_model  # (n_segments, 3, n_paths)
    M = weighted.reshape(
        n_segments * n_params_per_seg, n_paths
    ).T  # (n_paths, n_segments*3)

    logger.debug(
        "construct_forward_map: returning forward matrix M with shape %s", M.shape
    )

    return M


def construct_Cd(ref_phase: pd.Series, ic_tt: pd.Series) -> np.ndarray:
    """
    The noise levels for each reference phase are given in seconds, so we need to convert them to fractional traveltime perturbations by dividing by the inner core travel time.

    In principle this gives a different sigma for each observation.
    """

    noise_levels: dict[str, float] = {
        "ab": 0.95,
        "bc": 0.63,
        "cd": 0.29,
        "df": 0.95,
    }
    return np.diag((ref_phase.map(noise_levels) / ic_tt).astype(float).to_numpy())


register_builder("main:build_forward", make_builder(construct_forward_map))
register_builder("main:build_Cd", make_builder(construct_Cd))

# small helpers for priors
register_builder("main:eye", make_builder(lambda n_data: np.eye(n_data)))
register_builder(
    "main:prior_cov", make_builder(lambda n_params: 10.0 * np.eye(n_params))
)
register_builder("main:prior_mean", make_builder(lambda n_params: np.zeros(n_params)))
register_builder("main:noise_mean", make_builder(lambda n_data: np.zeros(n_data)))


ROOT = Path(__file__).parent.resolve()


def save(
    mean: np.ndarray, cov: np.ndarray, ev: float, outdir: Path | str | None = None
) -> Path:
    """Save the mean and covariance in numpy files, and the evidence in a txt files.

    outdir is an optional root output directory.  Output files will be saved in a timestamped subdirectory.

    Returns the full output directory path.
    """
    if outdir is None:
        outdir = ROOT / "outputs"
    outdir = Path(outdir)

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    hex = uuid.uuid4().hex[:8]
    outdir /= f"{now}_{hex}"
    outdir.mkdir(parents=True, exist_ok=False)
    np.save(outdir / "mean", mean)
    np.save(outdir / "cov", cov)
    with open(outdir / "evidence.txt", "w") as f:
        f.write(str(ev))
    return outdir


def main():
    """Run an inversion based on a given config file."""
    parser = argparse.ArgumentParser(
        description="Run IC anisotropy inference from a config file"
    )
    parser.add_argument(
        "--config", "-c", default=str(ROOT / "experiments" / "default.toml")
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))

    data_file = Path(cfg.data.file)
    logger.info("Reading data from %s", data_file)
    df = pd.read_parquet(data_file)
    data = (df.delta_t / df.inner_core_travel_time).astype(float).to_numpy()
    ic_in = np.stack(df.in_location.values)
    ic_out = np.stack(df.out_location.values)
    path_directions = calculate_path_direction_vector(ic_in, ic_out)

    logger.debug(
        "Loaded data: n_obs=%d ic_in.shape=%s ic_out.shape=%s",
        data.shape[0],
        ic_in.shape,
        ic_out.shape,
    )

    mesh = cfg.mesh.to_mesh()
    weights = determine_weights(mesh, ic_in, path_directions)

    context = {  # all the arguments to constructors only known at runtime
        "n_params": mesh.n_cells * 3,
        "n_data": data.size,
        "path_directions": path_directions,
        "weights": weights,
        "ref_phase": df.reference_phase,
        "ic_tt": df.inner_core_travel_time,
    }
    inferred, nuisance = cfg.components.to_gaussian_components(context=context)

    logger.info(
        "Running inference: prior cov shape=%s noise cov shape=%s",
        inferred[0].C.shape,
        nuisance[0].C.shape,
    )

    Cp = calc_posterior_cov(inferred, nuisance)
    logger.info("Posterior covariance shape: %s", Cp.shape)
    mp = calc_posterior_mean(data, inferred, nuisance)
    logger.info("Posterior mean shape: %s", mp.shape)
    Zp = calc_log_evidence(data, inferred, nuisance)
    logger.info("Log-evidence: %s", Zp)

    outdir = save(mp, Cp, Zp, cfg.output.prefix)

    # save resolved configuration for provenance
    resolved = {
        "config": config_to_json_dict(cfg),
        "derived": {
            "n_data": int(context["n_data"]),
            "n_params": int(context["n_params"]),
            "mesh_n_cells": getattr(mesh, "n_cells", None),
        },
    }
    save_resolved_config(outdir, resolved)
    logger.info("Saved outputs and resolved config to %s", outdir)


if __name__ == "__main__":
    main()
