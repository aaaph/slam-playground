from __future__ import annotations

from pathlib import Path

import pandas as pd
import rerun as rr

my_data_path = Path("./data.rrd")
server = rr.server.Server(datasets={"ds": [my_data_path]})
client = rr.catalog.CatalogClient(server.url())
dataset = client.get_dataset("ds")

ACCEL_BIAS_ENTITY = "/estimates/accel_bias/factor_graph"


def _accel_bias_table() -> pd.DataFrame:
    table = dataset.filter_contents(ACCEL_BIAS_ENTITY).reader(index="frame_time").to_pandas()
    scalar_col = f"{ACCEL_BIAS_ENTITY}:Scalars:scalars"
    if scalar_col not in table.columns:
        matches = [c for c in table.columns if c.endswith(":Scalars:scalars")]
        if not matches:
            msg = f"No :Scalars:scalars column for {ACCEL_BIAS_ENTITY}; got {list(table.columns)}"
            raise KeyError(msg)
        scalar_col = matches[0]

    bias_xyz = pd.DataFrame(table[scalar_col].tolist(), columns=["x", "y", "z"])
    return pd.concat([table[["frame_time"]].reset_index(drop=True), bias_xyz], axis=1)


pd_df_accel_bias_xyz = _accel_bias_table()
out_csv = Path("./accel_bias_xyz.csv")
pd_df_accel_bias_xyz.to_csv(out_csv, index=False)
print(out_csv.resolve())  # noqa: T201
