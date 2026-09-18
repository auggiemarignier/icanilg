"""Script to loop through a bunch of configurations."""

import logging
from itertools import product
from pathlib import Path

from joblib import Parallel, delayed
from linear_gaussian import clear_cache

from config.models import (
    ComponentsConfig,
    ComponentsDict,
    Config,
    DataConfig,
    MeshConfig,
    OutputConfig,
)

from .main import load_data, run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SHARED_DATA = None
SHARED_CONTEXT = None


def _init_worker(data_path: Path) -> None:
    global SHARED_DATA, SHARED_CONTEXT

    SHARED_DATA, SHARED_CONTEXT = load_data(data_path)


DATA_CFG = DataConfig(Path("data/brett2024_ic_traveltimes.parquet"))

noise_covar_kwargs = {
    "noise.block_iid": {},
    "noise.correlated_paths": {"corr_length": 5},
}
prior_covar_kwargs = {
    "prior.iid": {"scale": 0.1},
    "prior.spherically_correlated": {
        "sigma2": 0.1,
        "lat_corr_length": 30,
        "rad_corr_length": 300,
    },
}

def process(options: tuple[str, str, str, float, int, int]) -> None:
    """Configure and run."""
    run_id = "-".join(str(o) for o in options)
    fwd, noise_c, prior_c, ssiak, radres, latres = options
    mesh_cfg = MeshConfig(radial_resolution=radres, lateral_resolution=latres)
    output_cfg = OutputConfig("giant_experiment/ge")
    components: ComponentsDict = {
        "inferred": [
            {
                "A": {"builder": fwd, "kwargs": {"normalisation": 0.5}},
                "C": {"builder": prior_c, "kwargs": prior_covar_kwargs[prior_c]},
                "mu": {"builder": "prior.zero_mean", "kwargs": {}},
            }
        ],
        "nuisance": [
            {
                "A": {"builder": "forward.eye", "kwargs": {"scale": 1.0}},
                "C": {"builder": noise_c, "kwargs": noise_covar_kwargs[noise_c]},
                "mu": {"builder": "noise.zero_mean", "kwargs": {}},
            }
        ],
    }
    if ssiak != 0.0:
        components["inferred"].append(
            {
                "A": {"builder": "forward.ssi_ak_filter", "kwargs": {"radius": ssiak}},
                "C": {
                    "builder": "forward.ssi_ak_bias_cov",
                    "kwargs": {"scale": 0.05, "radius": ssiak},
                },
                "mu": {
                    "builder": "forward.ssi_ak_bias_mean",
                    "kwargs": {"radius": ssiak},
                },
            }
        )
    components_cfg = ComponentsConfig.from_dict(components)

    cfg = Config(DATA_CFG, mesh_cfg, components_cfg, output_cfg, {})
    run(SHARED_DATA, SHARED_CONTEXT, cfg, run_id, False)
    clear_cache()


if __name__ == "__main__":
    rad_res = list(range(1, 11))
    lat_res = [1] + list(range(10, 70, 10))
    forwards = [
        "forward.build_forward",
        "forward.build_rti_forward",
        "forward.build_iso_forward",
    ]
    noise_covars = ["noise.block_iid", "noise.correlated_paths"]
    prior_covars = ["prior.iid", "prior.spherically_correlated"]
    ssiak_rads = [0.0, 15.0]

    prod = product(forwards, noise_covars, prior_covars, ssiak_rads, rad_res, lat_res)
    Parallel(
        n_jobs=4,
        backend="multiprocessing",
        initializer=_init_worker,
        initargs=(str(DATA_CFG.file),),
    )(delayed(process)(options) for options in prod)
