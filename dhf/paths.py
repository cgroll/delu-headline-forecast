"""Project paths configuration.

All paths are resolved relative to the project root, making scripts runnable
from any working directory. Add a @property for each new data file introduced
in the pipeline.
"""

from pathlib import Path


class ProjPaths:
    """Centralized project paths.

    The root is inferred from the location of this file (pkg/), so scripts
    run correctly regardless of the working directory they are invoked from.
    """

    def __init__(self):
        self._pkg_path = Path(__file__).resolve().parent  # pkg/
        self._project_path = self._pkg_path.parent        # project root

    # ------------------------------------------------------------------ #
    # Top-level directories                                                #
    # ------------------------------------------------------------------ #

    @property
    def project_path(self) -> Path:
        """Root project directory."""
        return self._project_path

    @property
    def pkg_path(self) -> Path:
        """Source package directory (pkg/)."""
        return self._pkg_path

    @property
    def pipeline_path(self) -> Path:
        """Pipeline scripts directory."""
        return self._project_path / "pipeline"

    # ------------------------------------------------------------------ #
    # Data directories                                                     #
    # ------------------------------------------------------------------ #

    @property
    def data_path(self) -> Path:
        """Main data directory."""
        return self._project_path / "data"

    @property
    def downloads_path(self) -> Path:
        """Raw downloaded data."""
        return self.data_path / "downloads"

    @property
    def processed_data_path(self) -> Path:
        """Processed/transformed data."""
        return self.data_path / "processed"

    # ------------------------------------------------------------------ #
    # Output directories                                                   #
    # ------------------------------------------------------------------ #

    @property
    def output_path(self) -> Path:
        """Generated outputs root."""
        return self._project_path / "output"

    @property
    def images_path(self) -> Path:
        """Chart/figure images saved by pipeline scripts."""
        return self.output_path / "images"

    @property
    def reports_path(self) -> Path:
        """Report files."""
        return self.output_path / "reports"

    # ------------------------------------------------------------------ #
    # SMARD data files                                                     #
    # ------------------------------------------------------------------ #

    @property
    def smard_downloads_path(self) -> Path:
        """Raw SMARD series downloads directory."""
        return self.downloads_path / "smard"

    def smard_raw_file(self, name: str) -> Path:
        """Raw hourly SMARD series parquet.

        `name` is one of: pv, wind_onshore, wind_offshore, load.
        """
        return self.smard_downloads_path / f"{name}.parquet"

    # ------------------------------------------------------------------ #
    # External sibling projects                                            #
    # ------------------------------------------------------------------ #

    @property
    def mastr_project_path(self) -> Path:
        """Root of ~/research/mastr-power-capacities-germany, consumed directly
        per CLAUDE.md rather than re-derived from raw MaStR here.
        """
        return Path.home() / "research" / "mastr-power-capacities-germany"

    @property
    def mastr_nuts_regions_file(self) -> Path:
        """Germany NUTS region geometries (all levels), from GISCO. Used for
        map context (e.g. pipeline/09_pecd_zones_eda.py) — not for the
        MaStR capacity panels themselves, which live under `processed/`.
        """
        return self.mastr_project_path / "data" / "downloads" / "nuts_regions.geojson"

    @property
    def mastr_country_borders_file(self) -> Path:
        """Country-level outlines for Germany + North/Baltic Sea neighbors, for
        map context around offshore areas. See pipeline/09_pecd_zones_eda.py.
        """
        return self.mastr_project_path / "data" / "downloads" / "country_borders.geojson"

    @property
    def mastr_capacity_by_nuts2_month_file(self) -> Path:
        """Monthly installed capacity by NUTS2 region x series x month, from
        ~/research/mastr-power-capacities-germany. `series` values:
        solar_full_feed_in, solar_self_consumption_no_storage,
        solar_self_consumption_with_storage, solar_unknown, wind_onshore.
        See docs/data_sources.md.
        """
        return self.mastr_project_path / "data" / "processed" / "capacity_by_nuts2_month.parquet"

    @property
    def mastr_capacity_by_peon_month_file(self) -> Path:
        """Monthly onshore wind capacity by PEON zone (DE01..DE07) x month, from
        ~/research/mastr-power-capacities-germany's PEON/NUTS2 grid crosswalk.
        See docs/data_sources.md.
        """
        return self.mastr_project_path / "data" / "processed" / "capacity_by_peon_month.parquet"

    @property
    def mastr_capacity_by_peof_month_file(self) -> Path:
        """Monthly offshore wind capacity by PEOF zone (DE0xx_OFF) x month, from
        ~/research/mastr-power-capacities-germany's PEOF grid crosswalk. Zone
        codes already match this project's PECD offshore columns exactly
        (e.g. "DE011_OFF") — no relabeling needed to join. See
        docs/data_sources.md.
        """
        return self.mastr_project_path / "data" / "processed" / "capacity_by_peof_month.parquet"

    # ------------------------------------------------------------------ #
    # PECD data files                                                      #
    # ------------------------------------------------------------------ #

    @property
    def pecd_downloads_path(self) -> Path:
        """Raw PECD downloads directory."""
        return self.downloads_path / "pecd"

    @property
    def peon_mask_file(self) -> Path:
        """PEON (Pan-European Onshore wind zone) mask, from the CDS "weights and
        masks" widget: NetCDF with dims (region, latitude, longitude); `mask`
        gives each 0.25 degree grid cell's area-coverage fraction (0..1) for
        that zone, rasterized from ENTSO-E's zone shapefile. Germany
        intersects 7 zones: DE01..DE07. See pipeline/09_pecd_zones_eda.py.
        """
        return self.pecd_downloads_path / "peon_region_mask.nc"

    @property
    def peof_mask_file(self) -> Path:
        """PEOF (Pan-European Offshore wind zone) mask, same structure as
        `peon_mask_file`. Germany intersects 6 zones: DE011..DE015, DE02.
        """
        return self.pecd_downloads_path / "peof_region_mask.nc"

    @property
    def pecd_capacity_factors_downloads_path(self) -> Path:
        """Raw PECD capacity factor timeseries ZIPs directory."""
        return self.pecd_downloads_path / "capacity_factors"

    def pecd_capacity_factor_zip(self, kind: str, technology: str) -> Path:
        """Raw PECD capacity factor download for one (kind, technology) combo.

        `kind` is one of: solar, wind_onshore, wind_offshore. `technology` is
        the PECD technology code (e.g. "60" for SPV industrial rooftop, "30"
        for onshore "Existing technologies" — see docs/data_sources.md). Each
        ZIP holds one CSV per requested year (2015..2025), covering all
        German zones at that variable's spatial resolution (NUTS2 for solar,
        PEON/PEOF for wind). See pipeline/10_download_pecd_capacity_factors.py.
        """
        return self.pecd_capacity_factors_downloads_path / f"{kind}_tech{technology}.zip"

    def pecd_temperature_zip(self, file_version: str) -> Path:
        """Raw PECD 2m temperature download, country level (NUTS0), all of Europe.

        Not population-weighted — v4.2 only offers `population_weighted_temperature`
        at the pan-European "synchronous zone" resolution (Germany's zone,
        Continental Europe, spans ~20 countries — not usable for a
        Germany-specific signal), so this uses plain area-average
        `2m_temperature` at NUTS0 instead.

        `file_version` must be "fv1" (2022-2025 only) or "fv2" (1950-2021
        only) — this variable/resolution combo can't be requested in one
        call spanning both ranges (confirmed via the CDS constraints; a
        single fv1 request silently drops any pre-2022 years instead of
        erroring). See docs/data_sources.md and
        pipeline/14_download_pecd_temperature.py.
        """
        return self.pecd_downloads_path / f"temperature_de_{file_version}.zip"

    # ------------------------------------------------------------------ #
    # Processed data files                                                 #
    # ------------------------------------------------------------------ #

    @property
    def pecd_solar_capacity_factors_file(self) -> Path:
        """Hourly solar PV capacity factor, Germany-only NUTS2 regions.

        MultiIndex columns (technology, region): technology in {60, 61, 62,
        63} (industrial rooftop, residential rooftop, utility fixed, utility
        1-axis tracker — see docs/data_sources.md), region a German NUTS2
        code. See pipeline/11_process_pecd_capacity_factors.py.
        """
        return self.processed_data_path / "pecd_solar_capacity_factors.parquet"

    @property
    def pecd_wind_onshore_capacity_factors_file(self) -> Path:
        """Hourly onshore wind capacity factor ("Existing technologies"),
        columns = Germany's 7 PEON zones (DE01..DE07). See
        pipeline/11_process_pecd_capacity_factors.py.
        """
        return self.processed_data_path / "pecd_wind_onshore_capacity_factors.parquet"

    @property
    def pecd_wind_offshore_capacity_factors_file(self) -> Path:
        """Hourly offshore wind capacity factor ("Existing technologies"),
        columns = Germany's 6 PEOF zones (DE011_OFF..DE015_OFF, DE02_OFF).
        See pipeline/11_process_pecd_capacity_factors.py.
        """
        return self.processed_data_path / "pecd_wind_offshore_capacity_factors.parquet"

    @property
    def pecd_temperature_file(self) -> Path:
        """Hourly 2m temperature (degrees Celsius), Germany (NUTS0) only, single column "DE".

        See pipeline/15_process_pecd_temperature.py.
        """
        return self.processed_data_path / "pecd_temperature_de.parquet"

    @property
    def pecd_wind_zones_file(self) -> Path:
        """Derived vector polygons for Germany's PEON/PEOF wind zones.

        Built by thresholding+dissolving the raw grid-cell masks (see
        `peon_mask_file`/`peof_mask_file`) — one row per zone, columns
        `zone_id`, `scheme` (onshore/offshore), `geometry`. This is an
        approximation of the true (non-public) ENTSO-E shapefile at 0.25
        degree grid resolution, for visual sanity-checking and, if the zones
        look sensible, as a first grouping key for MaStR wind capacity
        (instead of NUTS2, which PECD v4.2 doesn't offer for wind — see
        docs/data_sources.md). See pipeline/09_pecd_zones_eda.py.
        """
        return self.processed_data_path / "pecd_wind_zones.geojson"

    @property
    def target_panel_file(self) -> Path:
        """Combined hourly DE-LU target panel (pv, wind_onshore, wind_offshore, load columns, MW)."""
        return self.processed_data_path / "target_panel.parquet"

    @property
    def results_path(self) -> Path:
        """Per-scenario metrics parquet files — see dhf/scenarios.py."""
        return self.processed_data_path / "results"

    def scenario_results_file(self, scenario_id: str) -> Path:
        """Per-(target, horizon) MAE/RMSE results + metadata for one scenario."""
        return self.results_path / f"{scenario_id}.parquet"

    # ------------------------------------------------------------------ #
    # Book                                                                 #
    # ------------------------------------------------------------------ #

    @property
    def book_notebooks_path(self) -> Path:
        """Executed notebooks directory consumed by the MyST book."""
        return self._project_path / "book" / "notebooks"

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def ensure_directories(self) -> None:
        """Create all standard directories if they do not yet exist."""
        dirs = [
            self.downloads_path,
            self.processed_data_path,
            self.images_path,
            self.reports_path,
            self.smard_downloads_path,
            self.pecd_downloads_path,
            self.pecd_capacity_factors_downloads_path,
            self.results_path,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
