from __future__ import annotations

import os

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pandas as pd


class MissingDataError(Exception):
    pass


class DataLoader:
    """Load one Roman light curve and convert magnitudes to fluxes."""

    def __init__(self, coord_file="data/coords.csv", filters=("F146", "F213", "F087")):
        self.coord_file = coord_file
        self.filters = filters
        self.event_coords = {}

        if os.path.exists(coord_file):
            df = pd.read_csv(coord_file)
            self.event_coords = {
                row["name"]: jnp.asarray(
                    [row["ra_deg"], row["dec_deg"]], dtype=jnp.float64
                )
                for _, row in df.iterrows()
            }
        else:
            print(
                f"Warning: coordinate file {coord_file!r} not found. "
                "Using [0, 0] fallback coords for models that do not need parallax."
            )

    def _find_existing_filter_file(self, file_path):
        if os.path.exists(file_path):
            return file_path
        for alt_filter in self.filters:
            alt_file = file_path.replace("F146", alt_filter)
            if os.path.exists(alt_file):
                return alt_file
        raise MissingDataError(f"No data found for {file_path}")

    def load_event(self, file_path):
        file_path = self._find_existing_filter_file(file_path)

        df = pd.read_csv(file_path, header=None, names=["bjd", "mag", "mag_err"])

        if len(df) == 0:
            for alt_filter in self.filters:
                alt_file = file_path.replace("F146", alt_filter)
                if os.path.exists(alt_file) and os.path.getsize(alt_file) > 0:
                    df = pd.read_csv(
                        alt_file, header=None, names=["bjd", "mag", "mag_err"]
                    )
                    if len(df) != 0:
                        file_path = alt_file
                        break
            else:
                raise MissingDataError(
                    f"No valid data found in filters {self.filters} for {file_path}"
                )

        df = df.apply(pd.to_numeric, errors="coerce").dropna()
        if len(df) == 0:
            raise MissingDataError(f"Only NaNs/non-numeric rows in {file_path}")

        # Important for initial-condition detection and stable trajectories.
        df = df.sort_values("bjd")

        flux = 10.0 ** (-0.4 * (df["mag"].values - 22.0))
        flux_err = flux * (0.4 * np.log(10.0)) * df["mag_err"].values
        jd = df["bjd"].values

        event_name = os.path.splitext(os.path.basename(file_path))[0]
        coords = self.event_coords.get(
            event_name, jnp.asarray([0.0, 0.0], dtype=jnp.float64)
        )

        return {
            "t": jnp.asarray(jd, dtype=jnp.float64),
            "mag": jnp.asarray(flux, dtype=jnp.float64),
            "mag_err": jnp.asarray(flux_err, dtype=jnp.float64),
            "coords": coords,
        }
