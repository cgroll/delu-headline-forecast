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
    # Processed data files                                                 #
    # ------------------------------------------------------------------ #

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
            self.results_path,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
