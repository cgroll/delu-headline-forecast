"""Download PECD v4.2 wind zone region masks (PEON onshore, PEOF offshore).

Pure data acquisition — no charts. PECD v4.2 does not offer NUTS-level
spatial aggregation for wind capacity factors, only two pan-European zone
schemes: PEON (onshore) and PEOF (offshore) — see docs/data_sources.md.
Germany intersects 7 PEON zones (DE01..DE07) and 6 PEOF zones (DE011..DE015,
DE02). Their boundaries aren't publicly available as a shapefile, but the
CDS "weights and masks" widget provides the same information rasterized: a
NetCDF per scheme giving each 0.25 degree grid cell's fractional area
coverage (0..1) for every European zone.

Used by pipeline/09_pecd_zones_eda.py to derive and sanity-check vector zone
polygons for Germany, before deciding whether to use them (instead of
NUTS2) to group MaStR wind capacity data.

Requires a ~/.cdsapirc file with a valid CDS API key.
"""

import zipfile

import cdsapi

from dhf.paths import ProjPaths

paths = ProjPaths()
paths.ensure_directories()

MASKS = {
    "peon": paths.peon_mask_file,
    "peof": paths.peof_mask_file,
}

client = cdsapi.Client()

for variable, output_file in MASKS.items():
    if output_file.exists():
        print(f"{variable}: already downloaded -> {output_file}")
        continue

    zip_path = output_file.with_suffix(".zip")
    request = {
        "pecd_version": "pecd4_2",
        "file_version": "fv1",
        "variable": f"{variable}_region_mask",
    }
    print(f"{variable}: requesting mask ...")
    client.retrieve("sis-energy-pecd", request, str(zip_path))

    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        assert len(names) == 1, f"expected a single file in {zip_path.name}, got {names}"
        with z.open(names[0]) as src, open(output_file, "wb") as dst:
            dst.write(src.read())
    zip_path.unlink()
    print(f"  saved -> {output_file}")
