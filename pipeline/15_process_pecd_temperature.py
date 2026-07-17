"""Process the downloaded PECD 2m temperature ZIPs into an hourly, Germany-only parquet.

Pure data processing — no charts. Same ZIP structure (one CSV per year
inside) and DE-column filtering as
pipeline/11_process_pecd_capacity_factors.py, reused directly rather than
duplicated. Combines the two file_version downloads (see
pipeline/14_download_pecd_temperature.py) into one continuous series.

Output: data/processed/pecd_temperature_de.parquet
  Hourly DatetimeIndex, single column "DE" (degrees Celsius).
"""

import pandas as pd

from dhf.paths import ProjPaths
from dhf.pecd import load_capacity_factor_zip

paths = ProjPaths()
paths.ensure_directories()

parts = [load_capacity_factor_zip(paths.pecd_temperature_zip(fv)) for fv in ("fv2", "fv1")]
temperature = pd.concat(parts).sort_index()
temperature = temperature[~temperature.index.duplicated(keep="last")]
print(f"{len(temperature):,} rows x {len(temperature.columns)} countries · {temperature.index.min()} .. {temperature.index.max()}")

de_temperature = temperature[["DE"]]
print(f"DE temperature range: {de_temperature['DE'].min():.1f}C .. {de_temperature['DE'].max():.1f}C")

de_temperature.to_parquet(paths.pecd_temperature_file)
print(f"Saved -> {paths.pecd_temperature_file}")
