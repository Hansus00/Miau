import os

import numpy as np
import pandas as pd

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


class MissingDataError(Exception):
    pass


class DataLoader:
    """Class for data loading and computing initial conditions."""

    def __init__(self, coord_file="data/coords.csv", filters=("F146", "F213", "F087")):
        self.coord_file = coord_file
        self.filters = filters
        df = pd.read_csv(coord_file)
        self.event_coords = {
            row["name"]: jnp.asarray([row["ra_deg"], row["dec_deg"]], dtype=jnp.float64)
            for _, row in df.iterrows()
        }

    def load_event(self, file_path):
        if not os.path.exists(file_path):
            found_alt = False
            for alt_filter in self.filters:
                alt_file = file_path.replace("F146", alt_filter)
                if os.path.exists(alt_file):
                    file_path = alt_file
                    found_alt = True
                    break
            if not found_alt:
                raise MissingDataError(f"No data found (or missing) for {file_path}")

        df = pd.read_csv(file_path, header=None, names=["bjd", "mag", "mag_err"])

        if len(df) == 0:
            for alt_filter in self.filters:
                alt_file = file_path.replace("F146", alt_filter)
                if os.path.exists(alt_file) and os.path.getsize(alt_file) > 0:
                    df = pd.read_csv(alt_file, header=None, names=["bjd", "mag", "mag_err"])
                    if len(df) != 0:
                        break
            else:
                raise MissingDataError(
                    f"No valid data found in filters {self.filters} for {file_path}"
                )

        flux = 10 ** (-0.4 * (df["mag"].values - 22)) # type: ignore
        flux_err = flux * (0.4 * np.log(10.0)) * df["mag_err"].values
        JD = df["bjd"].values

        event_name = os.path.splitext(os.path.basename(file_path))[0]

        return {
            "t": jnp.asarray(JD, dtype=jnp.float64),
            "mag": jnp.asarray(flux, dtype=jnp.float64),
            "mag_err": jnp.asarray(flux_err, dtype=jnp.float64),
            "coords": self.event_coords[event_name],
        }
