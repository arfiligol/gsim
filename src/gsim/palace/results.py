"""Load and map Palace S-parameter results to port names.

Standalone utility — works with a Palace output directory, or the results
dict returned by ``sim.run()``.

Usage::

    from gsim.palace.results import load_sparams

    sp = load_sparams(results)
    sp.plot()  # quick overview
    sp["o1", "o2"].db  # dB array for S(o1, o2)
    sp.s11  # shorthand for 2-port
    sp.freq  # frequency in GHz
"""

from __future__ import annotations

import json
import logging
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray

    from gsim.palace.mesh.postprocessing import PostprocessingIndexMap

logger = logging.getLogger(__name__)

_INDEXED_CSV_COLUMN_RE = re.compile(
    r"^(?P<prefix>[A-Za-z][A-Za-z0-9_]*)\[(?P<index>\d+)\](?P<suffix>.*)$"
)
_TERMINAL_MATRIX_COLUMN_RE = re.compile(r"\[i\]\[(?P<index>\d+)\]")
_ITERATION_DIR_RE = re.compile(r"iteration(\d+)$")
_EIG_CSV_RE = re.compile(r"eig.*\.csv$", re.IGNORECASE)
_CSV_INDEX_SECTIONS = {
    "domain-E.csv": "Domains.Postprocessing.Energy",
    "surface-Q.csv": "Boundaries.Postprocessing.Dielectric",
    "port-EPR.csv": "Boundaries.Postprocessing.SurfaceFlux",
}
_REPORT_SOURCE_COLUMNS = ("name", "path", "required", "present", "loaded", "message")
_DOMAIN_MATERIAL_COLUMNS = (
    "material_row_index",
    "material_attribute",
    "material_attributes",
    "domain_index",
    "section",
    "source_name",
    "physical_name",
    "entry_name",
    "role",
    "attributes",
    "metadata",
    "material_name",
    "permittivity",
    "loss_tangent",
    "conductivity",
    "permeability",
    "material_axes",
    "volume_name",
    "stack_material_name",
    "matched_material_name",
    "material_model_type",
    "material_model_source",
    "material_within_validity",
    "material_validity_note",
    "material_frequency_hz",
    "material_frequency_ghz",
    "raw_material_resolution",
    "raw_material",
)
_DIELECTRIC_INTERFACE_COLUMNS = (
    "interface_row_index",
    "surface_index",
    "surface_attribute",
    "surface_attributes",
    "section",
    "source_name",
    "physical_name",
    "entry_name",
    "role",
    "attributes",
    "metadata",
    "interface_type",
    "preset_name",
    "preset_source",
    "thickness",
    "permittivity",
    "loss_tangent",
    "interface_material_name",
    "matched_material_name",
    "material_model_type",
    "material_model_source",
    "material_within_validity",
    "material_validity_note",
    "material_frequency_hz",
    "material_frequency_ghz",
    "raw_material_resolution",
    "raw_interface",
)
_DOMAIN_LOSS_COLUMNS = (
    "row_index",
    "sample_column",
    "sample_value",
    "mode_index",
    "frequency_ghz",
    "domain_index",
    "section",
    "source_name",
    "physical_name",
    "entry_name",
    "role",
    "attributes",
    "E_elec_j",
    "E_mag_j",
    "p_elec",
    "p_mag",
    "material_attribute",
    "material_attributes",
    "material_name",
    "material_permittivity",
    "material_loss_tangent",
    "material_conductivity",
    "material_permeability",
    "material_axes",
    "loss_tangent",
    "inverse_q",
    "q_equivalent",
    "gamma_rad_per_s",
    "gamma_per_us",
    "gamma_hz",
    "gamma_mhz",
    "t1_us",
)
_SURFACE_LOSS_COLUMNS = (
    "row_index",
    "sample_column",
    "sample_value",
    "mode_index",
    "frequency_ghz",
    "surface_index",
    "section",
    "source_name",
    "physical_name",
    "entry_name",
    "role",
    "attributes",
    "interface_type",
    "preset_name",
    "preset_source",
    "p_surf",
    "q_surf",
    "inverse_q",
    "q_equivalent",
    "surface_attribute",
    "surface_attributes",
    "thickness",
    "permittivity",
    "loss_tangent",
    "gamma_rad_per_s",
    "gamma_per_us",
    "gamma_hz",
    "gamma_mhz",
    "t1_us",
)
_LOSS_BUDGET_COLUMNS = (
    "mode_index",
    "frequency_ghz",
    "q_eig",
    "inverse_q_eig",
    "domain_inverse_q_sum",
    "surface_inverse_q_sum",
    "total_inverse_q_sum",
    "eig_with_surface_inverse_q_sum",
    "q_total",
    "q_eig_with_surface",
    "domain_vs_eig_relative_error",
    "gamma_rad_per_s",
    "gamma_per_us",
    "gamma_hz",
    "gamma_mhz",
    "t1_us",
)
_TERMINAL_MATRIX_SPECS = {
    "C": {
        "file_name": "terminal-C.csv",
        "source_unit": "F",
        "display_scale": 1.0e15,
        "display_unit": "fF",
    },
    "Cm": {
        "file_name": "terminal-Cm.csv",
        "source_unit": "F",
        "display_scale": 1.0e15,
        "display_unit": "fF",
    },
    "Cinv": {
        "file_name": "terminal-Cinv.csv",
        "source_unit": "1/F",
        "display_scale": 1.0,
        "display_unit": "1/F",
    },
}


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------


class SParam:
    """A single S-parameter entry (complex-valued vs frequency)."""

    def __init__(self, db: NDArray, deg: NDArray) -> None:
        """Create from dB magnitude and degree phase arrays."""
        self._db = db
        self._deg = deg

    @property
    def db(self) -> NDArray:
        """Magnitude in dB."""
        return self._db

    @property
    def deg(self) -> NDArray:
        """Phase in degrees."""
        return self._deg

    @property
    def mag(self) -> NDArray:
        """Linear magnitude."""
        return 10 ** (self._db / 20)

    @property
    def complex(self) -> NDArray:
        """Complex S-parameter values."""
        return self.mag * np.exp(1j * np.deg2rad(self._deg))

    def __repr__(self) -> str:
        """Return string representation."""
        return f"SParam(n={len(self._db)})"


class SParams:
    """Palace S-parameter results with named port access.

    Access individual S-parameters by port name pair::

        sp["o1", "o2"]  # -> SParam object
        sp["o1", "o2"].db  # -> dB array
        sp["o1", "o2"].deg  # -> phase array

    For 2-port convenience, RF shorthand works::

        sp.s11  # sp[ports[0], ports[0]]
        sp.s21  # sp[ports[1], ports[0]]
        sp.s12  # sp[ports[0], ports[1]]
        sp.s22  # sp[ports[1], ports[1]]
    """

    def __init__(
        self,
        freq: NDArray,
        data: dict[tuple[str, str], SParam],
        port_names: list[str],
        files: dict[str, Path] | None = None,
    ) -> None:
        """Create from frequency array, S-parameter data, and port names."""
        self._freq = freq
        self._data = data
        self._port_names = port_names
        self.files = files or {}

    @property
    def freq(self) -> NDArray:
        """Frequency in GHz."""
        return self._freq

    @property
    def port_names(self) -> list[str]:
        """Ordered list of port names."""
        return list(self._port_names)

    def __getitem__(self, key: tuple[str, str]) -> SParam:
        """Get S-parameter by port name pair: sp["o1", "o2"]."""
        if not isinstance(key, tuple) or len(key) != 2:
            msg = f'Use sp["to", "from"] indexing, got {key!r}'
            raise KeyError(msg)
        if key not in self._data:
            available = [f'("{k[0]}", "{k[1]}")' for k in sorted(self._data)]
            msg = (
                f'S-parameter ("{key[0]}", "{key[1]}") not found. '
                f"Available: {', '.join(available)}"
            )
            raise KeyError(msg)
        return self._data[key]

    def __getattr__(self, name: str) -> SParam:
        """RF shorthand: sp.s11, sp.s21, etc."""
        m = re.fullmatch(r"s(\d)(\d)", name)
        if m and len(self._port_names) >= 2:
            i, j = int(m.group(1)), int(m.group(2))
            if 1 <= i <= len(self._port_names) and 1 <= j <= len(self._port_names):
                to_port = self._port_names[i - 1]
                from_port = self._port_names[j - 1]
                key = (to_port, from_port)
                if key in self._data:
                    return self._data[key]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def keys(self) -> list[tuple[str, str]]:
        """All available (to_port, from_port) pairs."""
        return list(self._data.keys())

    def to_dataframe(self):
        """Export to a flat pandas DataFrame."""
        import pandas as pd

        cols: dict[str, NDArray] = {"freq_ghz": self._freq}
        for (to_p, from_p), sp in self._data.items():
            cols[f"S_{to_p}_{from_p}_db"] = sp.db
            cols[f"S_{to_p}_{from_p}_deg"] = sp.deg
        return pd.DataFrame(cols)

    def _filtered_entries(self, full: bool) -> list[tuple[str, SParam]]:
        """Return ``[(label, SParam), ...]`` filtered by excitation port."""
        from_ports = list(dict.fromkeys(fp for _, fp in self._data))
        first_from = from_ports[0] if from_ports else None
        entries: list[tuple[str, SParam]] = []
        for (to_p, from_p), sp in self._data.items():
            if not full and len(from_ports) > 1 and from_p != first_from:
                continue
            entries.append((f"S({to_p},{from_p})", sp))
        return entries

    def plot(
        self,
        *,
        full: bool = False,
        figsize: tuple[float, float] = (8, 6),
    ) -> None:
        """Plot magnitude and phase with matplotlib (static).

        By default only the first excitation column is plotted
        (e.g. S11, S21 but not S12, S22). Pass ``full=True`` to
        include all entries.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        for label, sp in self._filtered_entries(full):
            ax1.plot(self._freq, sp.db, label=label)
            ax2.plot(self._freq, sp.deg, label=label)

        ax1.set_ylabel("Magnitude (dB)")
        ax1.set_title("S-Parameters")
        ax1.legend()
        ax1.grid(True)

        ax2.set_xlabel("Frequency (GHz)")
        ax2.set_ylabel("Phase (deg)")
        ax2.legend()
        ax2.grid(True)

        fig.tight_layout()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.show()

    def plot_plotly(self, *, full: bool = False):
        """Plot S-parameters with Plotly (interactive).

        Returns a ``plotly.graph_objects.Figure`` that renders interactively
        in notebooks and can be saved as standalone HTML via
        ``fig.write_html("sparams.html")``.
        """
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("Magnitude (dB)", "Phase (deg)"),
        )
        for label, sp in self._filtered_entries(full):
            fig.add_scatter(
                x=self._freq,
                y=sp.db,
                mode="lines",
                name=label,
                legendgroup=label,
                row=1,
                col=1,
            )
            fig.add_scatter(
                x=self._freq,
                y=sp.deg,
                mode="lines",
                name=label,
                legendgroup=label,
                showlegend=False,
                row=2,
                col=1,
            )
        fig.update_xaxes(title_text="Frequency (GHz)", row=2, col=1)
        fig.update_yaxes(title_text="dB", row=1, col=1)
        fig.update_yaxes(title_text="deg", row=2, col=1)
        fig.update_layout(title="S-Parameters", height=600)
        return fig

    def _port_index_map(self) -> dict[str, int]:
        """Return ``{port_name: 1-based index}`` mapping."""
        return {name: i + 1 for i, name in enumerate(self._port_names)}

    def _sij_label(self, to_port: str, from_port: str) -> str:
        """Return ``Sij`` label for a port pair."""
        idx = self._port_index_map()
        return f"S{idx[to_port]}{idx[from_port]}"

    def plot_interactive(self, phase: bool = False):
        """Plot S-parameters with interactive legend toggling.

        Uses ``Sij`` notation (e.g. S11, S21) and prints the port
        mapping so you know which index corresponds to which port.
        By default shows the first excitation column (S11, S21, S31, ...)
        and hides symmetric/redundant entries. All traces are togglable
        via the legend.

        Args:
            phase: If True, plot phase (deg). Default is magnitude (dB).

        Returns:
            plotly Figure
        """
        import plotly.graph_objects as go  # type: ignore[import-untyped]

        # Print port mapping
        idx = self._port_index_map()
        mapping = ", ".join(f"Port {i}: {name}" for name, i in idx.items())
        print(f"Port mapping: {mapping}")  # noqa: T201

        # Build entries with Sij labels
        entries: list[tuple[str, str, SParam]] = []
        for (to_p, from_p), sp in self._data.items():
            entries.append((self._sij_label(to_p, from_p), from_p, sp))

        # Show first excitation column by default (Si1), hide the rest
        first_from = self._port_names[0] if self._port_names else None
        first_col = [(l, sp) for l, fp, sp in entries if fp == first_from]
        rest = [(l, sp) for l, fp, sp in entries if fp != first_from]
        ordered = first_col + rest
        visible_set = {l for l, _ in first_col}

        fig = go.Figure()

        for label, sp in ordered:
            y = sp.deg if phase else sp.db
            vis = True if label in visible_set else "legendonly"
            fig.add_scatter(
                x=self._freq,
                y=y,
                mode="lines",
                name=label,
                visible=vis,
            )

        ylabel = "Phase (deg)" if phase else "|S| (dB)"
        fig.update_layout(
            xaxis_title="Frequency (GHz)",
            yaxis_title=ylabel,
            width=650,
            height=350,
            margin=dict(t=40, b=40, l=60, r=140),
            modebar=dict(orientation="v"),
            legend=dict(
                x=1.02,
                y=1,
                xanchor="left",
                yanchor="top",
                groupclick="toggleitem",
                itemclick="toggle",
                itemdoubleclick="toggleothers",
                itemsizing="constant",
                bordercolor="#888",
                borderwidth=1,
                bgcolor="rgba(245,245,245,0.9)",
                entrywidthmode="pixels",
                entrywidth=70,
            ),
        )
        return fig

    def save_npz(self, filepath: str | Path) -> Path:
        """Save S-parameters to a ``.npz`` file.

        The file can be reloaded with :meth:`SParams.from_file`.

        Args:
            filepath: Destination path (``.npz`` suffix added if missing).

        Returns:
            The resolved file path.
        """
        filepath = Path(filepath).with_suffix(".npz")
        filepath.parent.mkdir(parents=True, exist_ok=True)

        arrays: dict[str, NDArray] = {"freq": self._freq}
        arrays["port_names"] = np.array(self._port_names)
        for (to_p, from_p), sp in self._data.items():
            arrays[f"S_{to_p}_{from_p}_db"] = sp.db
            arrays[f"S_{to_p}_{from_p}_deg"] = sp.deg

        np.savez_compressed(str(filepath), **arrays)  # pyright: ignore[reportArgumentType]
        logger.info("S-parameters saved to %s", filepath)
        return filepath

    @classmethod
    def from_file(cls, filepath: str | Path) -> SParams:
        """Load S-parameters from a ``.npz`` file written by :meth:`save_npz`.

        Args:
            filepath: Path to the ``.npz`` file.

        Returns:
            Reconstructed :class:`SParams` object.
        """
        filepath = Path(filepath).with_suffix(".npz")
        npz = np.load(filepath, allow_pickle=False)

        freq = npz["freq"]
        port_names = list(npz["port_names"])

        data: dict[tuple[str, str], SParam] = {}
        for to_p in port_names:
            for from_p in port_names:
                db_key = f"S_{to_p}_{from_p}_db"
                deg_key = f"S_{to_p}_{from_p}_deg"
                if db_key in npz and deg_key in npz:
                    data[(to_p, from_p)] = SParam(db=npz[db_key], deg=npz[deg_key])

        logger.info("S-parameters loaded from %s", filepath)
        return cls(freq=freq, data=data, port_names=port_names)

    def __repr__(self) -> str:
        """Return string representation."""
        n_freq = len(self._freq)
        n_ports = len(self._port_names)
        ports_str = ", ".join(self._port_names)
        return (
            f"SParams({n_ports} ports [{ports_str}], "
            f"{n_freq} freq points, "
            f"{len(self._data)} S-parameters)"
        )


@dataclass(frozen=True)
class IndexedCsvColumn:
    """Mapping from one Palace indexed CSV column to mesh identity."""

    original_name: str
    renamed_name: str
    quantity: str
    index: int
    section: str
    physical_name: str | None = None
    entry_name: str | None = None
    role: str | None = None
    attributes: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly audit row."""
        row: dict[str, Any] = {
            "original_name": self.original_name,
            "renamed_name": self.renamed_name,
            "quantity": self.quantity,
            "index": self.index,
            "section": self.section,
        }
        if self.physical_name is not None:
            row["physical_name"] = self.physical_name
        if self.entry_name is not None:
            row["entry_name"] = self.entry_name
        if self.role is not None:
            row["role"] = self.role
        if self.attributes:
            row["attributes"] = list(self.attributes)
        if self.metadata:
            row["metadata"] = dict(self.metadata)
        if self.extra:
            row["extra"] = dict(self.extra)
        return row


@dataclass(frozen=True)
class IndexedCsv:
    """Palace indexed CSV with optional physical-name column labels."""

    source_path: Path
    section: str
    dataframe: pd.DataFrame
    columns: tuple[IndexedCsvColumn, ...]

    @property
    def column_map(self) -> tuple[dict[str, Any], ...]:
        """Return JSON-friendly column provenance rows."""
        return tuple(column.to_dict() for column in self.columns)


@dataclass(frozen=True)
class Eigenmodes:
    """Palace Eigenmode result table with normalized report columns."""

    source_path: Path
    dataframe: pd.DataFrame

    @property
    def n_modes(self) -> int:
        """Number of loaded eigenmodes."""
        return len(self.dataframe)

    @property
    def mode_indices(self) -> NDArray:
        """Palace mode indices as integers."""
        return self.dataframe["mode_index"].to_numpy(dtype=int)

    @property
    def freq_real_ghz(self) -> NDArray:
        """Real part of the complex eigenfrequency in GHz."""
        return self.dataframe["freq_real_ghz"].to_numpy(dtype=float)

    @property
    def freq_imag_ghz(self) -> NDArray:
        """Imaginary part of the complex eigenfrequency in GHz."""
        return self.dataframe["freq_imag_ghz"].to_numpy(dtype=float)

    @property
    def q(self) -> NDArray:
        """Palace-reported quality factor values."""
        return self.dataframe["q"].to_numpy(dtype=float)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a copy of the normalized eigenmode table."""
        frame = self.dataframe.copy()
        frame.attrs.update({"csv_path": str(self.source_path)})
        return frame

    def to_report_dataframe(self) -> pd.DataFrame:
        """Return notebook-facing column aliases for eigenmode reports."""
        frame = self.to_dataframe().rename(
            columns={
                "freq_real_ghz": "frequency_ghz",
                "freq_imag_ghz": "imaginary_frequency_ghz",
                "q": "q_factor",
                "error_backward": "backward_error",
                "error_absolute": "absolute_error",
            }
        )
        return cast("pd.DataFrame", frame)


@dataclass(frozen=True)
class DrivenReport:
    """Composed Palace Driven report tables for notebook workflows."""

    sparams: SParams
    port_epr: pd.DataFrame
    domain_materials: pd.DataFrame
    dielectric_interfaces: pd.DataFrame
    index_map: pd.DataFrame
    sources: pd.DataFrame

    @property
    def network(self) -> SParams:
        """Return the driven S-parameter network."""
        return self.sparams

    @property
    def missing_reports(self) -> tuple[str, ...]:
        """Optional report names that were expected but absent."""
        if self.sources.empty:
            return ()
        missing = self.sources.loc[
            (~self.sources["required"]) & (~self.sources["present"]),
            "name",
        ]
        return tuple(str(name) for name in missing)


@dataclass(frozen=True)
class EigenmodeReport:
    """Composed Palace Eigenmode report tables for notebook workflows."""

    eigenmodes: Eigenmodes
    mode_history: pd.DataFrame
    pass_summary: pd.DataFrame
    domain_materials: pd.DataFrame
    dielectric_interfaces: pd.DataFrame
    domain_energy: pd.DataFrame
    domain_loss: pd.DataFrame
    surface_q: pd.DataFrame
    surface_loss: pd.DataFrame
    surface_interface_summary: pd.DataFrame
    loss_budget: pd.DataFrame
    port_epr: pd.DataFrame
    index_map: pd.DataFrame
    sources: pd.DataFrame

    @property
    def modes(self) -> pd.DataFrame:
        """Return final mode rows with notebook-facing column names."""
        return self.eigenmodes.to_report_dataframe()

    @property
    def missing_reports(self) -> tuple[str, ...]:
        """Optional report names that were expected but absent."""
        if self.sources.empty:
            return ()
        missing = self.sources.loc[
            (~self.sources["required"])
            & (~self.sources["present"])
            & (self.sources["name"] != "iteration*/eig.csv"),
            "name",
        ]
        return tuple(str(name) for name in missing)


@dataclass(frozen=True)
class ElectrostaticReport:
    """Composed Palace Electrostatic report tables for notebook workflows."""

    terminal_c: TerminalMatrix
    terminal_cm: TerminalMatrix | None
    terminal_cinv: TerminalMatrix | None
    terminal_c_history: pd.DataFrame
    terminal_cm_history: pd.DataFrame
    terminal_cinv_history: pd.DataFrame
    terminal_c_pass_summary: pd.DataFrame
    terminal_cm_pass_summary: pd.DataFrame
    terminal_cinv_pass_summary: pd.DataFrame
    domain_materials: pd.DataFrame
    dielectric_interfaces: pd.DataFrame
    domain_energy: pd.DataFrame
    domain_loss: pd.DataFrame
    surface_q: pd.DataFrame
    surface_loss: pd.DataFrame
    surface_interface_summary: pd.DataFrame
    loss_budget: pd.DataFrame
    index_map: pd.DataFrame
    sources: pd.DataFrame

    @property
    def capacitance(self) -> TerminalMatrix:
        """Return the capacitance matrix report."""
        return self.terminal_c

    @property
    def mutual_capacitance(self) -> TerminalMatrix | None:
        """Return the mutual capacitance matrix report when present."""
        return self.terminal_cm

    @property
    def inverse_capacitance(self) -> TerminalMatrix | None:
        """Return the inverse capacitance matrix report when present."""
        return self.terminal_cinv

    @property
    def matrix_history(self) -> dict[str, pd.DataFrame]:
        """Return terminal matrix convergence tables by matrix kind."""
        return {
            "C": self.terminal_c_history,
            "Cm": self.terminal_cm_history,
            "Cinv": self.terminal_cinv_history,
        }

    @property
    def matrix_pass_summary(self) -> dict[str, pd.DataFrame]:
        """Return terminal matrix pass summaries by matrix kind."""
        return {
            "C": self.terminal_c_pass_summary,
            "Cm": self.terminal_cm_pass_summary,
            "Cinv": self.terminal_cinv_pass_summary,
        }

    @property
    def missing_reports(self) -> tuple[str, ...]:
        """Optional report names that were expected but absent."""
        if self.sources.empty:
            return ()
        missing = self.sources.loc[
            (~self.sources["required"])
            & (~self.sources["present"])
            & (~self.sources["name"].str.startswith("iteration*/")),
            "name",
        ]
        return tuple(str(name) for name in missing)


@dataclass(frozen=True)
class TerminalMatrix:
    """Palace electrostatic terminal matrix with named terminals."""

    source_path: Path
    matrix_kind: str
    dataframe: pd.DataFrame
    terminal_names: tuple[str, ...]
    source_unit: str
    display_scale: float
    display_unit: str

    @property
    def display_dataframe(self) -> pd.DataFrame:
        """Return a copy scaled for notebook display."""
        frame = self.dataframe * self.display_scale
        frame.attrs.update(
            {
                "matrix_kind": self.matrix_kind,
                "source_unit": self.source_unit,
                "display_scale": self.display_scale,
                "display_unit": self.display_unit,
                "csv_path": str(self.source_path),
            }
        )
        return frame

    def to_long_dataframe(self) -> pd.DataFrame:
        """Return a long-form terminal-pair table."""
        import pandas as pd

        rows: list[dict[str, Any]] = []
        for row_offset, row_name in enumerate(self.terminal_names, start=1):
            for col_offset, col_name in enumerate(self.terminal_names, start=1):
                value = float(self.dataframe.loc[row_name, col_name])
                rows.append(
                    {
                        "matrix_kind": self.matrix_kind,
                        "matrix_csv_path": str(self.source_path),
                        "row_index": row_offset,
                        "column_index": col_offset,
                        "row_terminal": row_name,
                        "column_terminal": col_name,
                        "element": f"{row_name} -> {col_name}",
                        "is_diagonal": row_offset == col_offset,
                        "value_si": value,
                        "source_unit": self.source_unit,
                        "display_value": value * self.display_scale,
                        "display_unit": self.display_unit,
                        "display_scale": self.display_scale,
                    }
                )
        return pd.DataFrame.from_records(rows)


def load_sparams(
    source: str | Path | dict,
    *,
    port_info_path: str | Path | None = None,
) -> SParams:
    """Load Palace S-parameter results.

    Args:
        source: One of:

            - **results dict** returned by ``sim.run()``
            - **directory path** — the sim dir or ``output/palace/``
        port_info_path: Explicit path to ``port_information.json``.

    Returns:
        :class:`SParams` object with named port access.

    Raises:
        FileNotFoundError: If ``port-S.csv`` cannot be found.
    """
    import pandas as pd

    csv_path, base_dir = _resolve_source(source)
    if csv_path is None:  # pragma: no cover — _resolve_source raises first
        msg = "port-S.csv not found"
        raise FileNotFoundError(msg)

    # Check results dict for port_information.json (injected by DrivenSim)
    if port_info_path is None and isinstance(source, dict):
        pi_val = source.get("port_information.json")
        if pi_val is not None:
            port_info_path = Path(pi_val)

    port_map = _load_port_map(base_dir, csv_path, port_info_path)

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Extract frequency
    freq_col = next((c for c in df.columns if c.startswith("f")), None)
    freq = df[freq_col].to_numpy() if freq_col else np.arange(len(df))

    # Parse S-parameter columns into SParam objects
    # Group by (i, j) pair — each pair has a dB and deg column
    raw: dict[tuple[int, int], dict[str, NDArray]] = {}
    for col in df.columns:
        parsed = _parse_sparam_col(col)
        if parsed is None:
            continue
        i, j, kind = parsed
        raw.setdefault((i, j), {})[kind] = df[col].to_numpy()

    # Build port name list (ordered by index)
    all_indices = set()
    for i, j in raw:
        all_indices.add(i)
        all_indices.add(j)
    port_names = [port_map.get(idx, f"p{idx}") for idx in sorted(all_indices)]

    # Build SParam objects keyed by (port_name, port_name)
    data: dict[tuple[str, str], SParam] = {}
    for (i, j), parts in sorted(raw.items()):
        to_name = port_map.get(i, f"p{i}")
        from_name = port_map.get(j, f"p{j}")
        db = parts.get("db", np.zeros(len(freq)))
        deg = parts.get("deg", np.zeros(len(freq)))
        data[(to_name, from_name)] = SParam(db=db, deg=deg)

    files = dict(source) if isinstance(source, dict) else None
    return SParams(freq=freq, data=data, port_names=port_names, files=files)


def load_driven_report(
    source: str | Path | dict,
    *,
    index_map_path: str | Path | None = None,
    port_info_path: str | Path | None = None,
    require_port_epr: bool = False,
) -> DrivenReport:
    """Load Driven S-parameters plus optional indexed port-EPR reports.

    This is a thin composition layer over the stricter primitive loaders. The
    final ``port-S.csv`` is required. Palace index/config/provenance artifacts
    and ``port-EPR.csv`` are loaded independently when present and are reported
    as missing rather than forcing every Driven run to emit all report families.
    """
    import pandas as pd

    sparams = load_sparams(source, port_info_path=port_info_path)
    port_s_path, _ = _resolve_source(source)
    source_rows: list[dict[str, Any]] = [
        _report_source_row(
            "port-S.csv",
            port_s_path,
            required=True,
            present=True,
            loaded=True,
            message="loaded driven S-parameters",
        )
    ]

    resolved_index_map_path = _find_optional_postprocessing_index_map_path(
        source,
        index_map_path=index_map_path,
    )
    index_map_present = (
        resolved_index_map_path is not None and resolved_index_map_path.exists()
    )
    index_map_loaded = False
    if index_map_present:
        index_map = load_postprocessing_index_map(
            source,
            index_map_path=resolved_index_map_path,
        )
        index_map_frame = _postprocessing_index_map_to_dataframe(index_map)
        index_map_loaded = True
        index_message = "loaded postprocessing index map"
    else:
        index_map_frame = _empty_index_map_dataframe()
        index_message = "not found"
    source_rows.append(
        _report_source_row(
            "palace_index_map.json",
            resolved_index_map_path,
            required=False,
            present=index_map_present,
            loaded=index_map_loaded,
            message=index_message,
        )
    )

    resolved_config_path = _find_optional_config_path(source, config_path=None)
    config_present = resolved_config_path is not None and resolved_config_path.exists()
    config_loaded = False
    if config_present:
        domain_materials = load_domain_material_summary(
            source,
            config_path=resolved_config_path,
            index_map_path=resolved_index_map_path if index_map_present else None,
        )
        dielectric_interfaces = load_dielectric_interface_summary(
            source,
            config_path=resolved_config_path,
            index_map_path=resolved_index_map_path if index_map_present else None,
        )
        config_loaded = True
        config_message = (
            "loaded config material and interface summaries"
            if index_map_present
            else (
                "loaded config material and interface summaries without "
                "palace_index_map.json"
            )
        )
    else:
        domain_materials = _empty_domain_material_summary()
        dielectric_interfaces = _empty_dielectric_interface_summary()
        config_message = "not found"
    source_rows.append(
        _report_source_row(
            "config.json",
            resolved_config_path,
            required=False,
            present=config_present,
            loaded=config_loaded,
            message=config_message,
        )
    )

    port_epr, port_source = _load_optional_eigenmode_report_table(
        source,
        "port-EPR.csv",
        loader=load_port_epr_summary,
        empty_factory=_empty_port_epr_summary,
        index_map_path=resolved_index_map_path,
        index_map_present=index_map_present,
    )
    source_rows.append(port_source)

    if require_port_epr and not bool(port_source["loaded"]):
        msg = "Missing required driven port-EPR.csv report"
        raise FileNotFoundError(msg)

    return DrivenReport(
        sparams=sparams,
        port_epr=port_epr,
        domain_materials=domain_materials,
        dielectric_interfaces=dielectric_interfaces,
        index_map=index_map_frame,
        sources=pd.DataFrame.from_records(
            source_rows,
            columns=_REPORT_SOURCE_COLUMNS,
        ),
    )


def load_eigenmodes(source: str | Path | dict) -> Eigenmodes:
    """Load Palace ``eig.csv`` as a normalized Eigenmode result table.

    Args:
        source: CSV path, simulation directory, Palace output directory, or
            results dict returned by ``EigenmodeSim.run()`` / ``run_local()``.

    Returns:
        :class:`Eigenmodes` with stable columns:
        ``mode_index``, ``freq_real_ghz``, ``freq_imag_ghz``, ``q``,
        ``error_backward``, and ``error_absolute``.
    """
    import pandas as pd

    csv_path = _resolve_eig_csv(source)
    raw = pd.read_csv(csv_path, skipinitialspace=True)
    raw.columns = [str(column).strip() for column in raw.columns]
    if raw.empty:
        msg = f"Palace eigenmode CSV is empty: {csv_path}"
        raise ValueError(msg)
    columns = [str(column) for column in raw.columns]

    mode_col = _find_eigenmode_column(columns, ("m", "mode"), default=columns[0])
    real_col = _find_eigenmode_column(
        columns,
        ("re{f", "re(f", "real"),
        default=columns[1] if len(columns) > 1 else None,
    )
    imag_col = _find_eigenmode_column(
        columns,
        ("im{f", "im(f", "imag"),
        default=columns[2] if len(columns) > 2 else None,
    )
    q_col = _find_eigenmode_column(columns, ("q",), default=None)
    backward_error_col = _find_eigenmode_column(
        columns,
        ("bkwd", "backward"),
        default=None,
    )
    absolute_error_col = _find_eigenmode_column(
        columns,
        ("abs", "absolute"),
        default=None,
    )

    if real_col is None:
        msg = f"Palace eigenmode CSV has no real-frequency column: {csv_path}"
        raise ValueError(msg)

    frame = pd.DataFrame(
        {
            "mode_index": raw[mode_col].astype(float).round().astype(int),
            "freq_real_ghz": raw[real_col].astype(float),
            "freq_imag_ghz": (
                raw[imag_col].astype(float)
                if imag_col is not None
                else np.zeros(len(raw), dtype=float)
            ),
            "q": (
                raw[q_col].astype(float)
                if q_col is not None
                else np.full(len(raw), np.nan)
            ),
            "error_backward": (
                raw[backward_error_col].astype(float)
                if backward_error_col is not None
                else np.full(len(raw), np.nan)
            ),
            "error_absolute": (
                raw[absolute_error_col].astype(float)
                if absolute_error_col is not None
                else np.full(len(raw), np.nan)
            ),
        }
    )
    frame.attrs.update(
        {
            "csv_path": str(csv_path),
            "source_columns": tuple(columns),
        }
    )
    return Eigenmodes(source_path=csv_path, dataframe=frame)


def load_eigenmode_history(
    source: str | Path,
    *,
    include_final: bool = True,
) -> pd.DataFrame:
    """Load Palace Eigenmode AMR history from ``iteration*/eig.csv`` files.

    ``source`` may be a simulation directory or the ``output/palace`` directory.
    The returned frame includes source visibility columns and convergence
    deltas suitable for notebook reports.
    """
    import pandas as pd

    base = Path(source)
    if base.is_file():
        msg = "load_eigenmode_history() expects a simulation/output directory."
        raise ValueError(msg)

    output_dir = _resolve_palace_output_dir(base)
    pass_frames: list[pd.DataFrame] = []
    last_table: pd.DataFrame | None = None

    for iteration_dir, iteration_index in _iteration_dirs(output_dir):
        csv_path = iteration_dir / "eig.csv"
        if not csv_path.exists():
            continue
        modes = load_eigenmodes(csv_path)
        pass_frames.append(
            _eigenmodes_to_history_frame(
                modes,
                iteration_index=iteration_index,
                label=f"Pass {iteration_index}",
                is_final=False,
                source_kind="iteration",
                source_iteration=iteration_index,
            )
        )
        last_table = modes.dataframe

    if include_final:
        final_csv_path = output_dir / "eig.csv"
        if final_csv_path.exists():
            final_modes = load_eigenmodes(final_csv_path)
            if last_table is None or not _eigenmode_tables_match(
                last_table,
                final_modes.dataframe,
            ):
                iteration_index = (
                    1 if not pass_frames else _next_eigenmode_iteration(pass_frames)
                )
                pass_frames.append(
                    _eigenmodes_to_history_frame(
                        final_modes,
                        iteration_index=iteration_index,
                        label="Final",
                        is_final=True,
                        source_kind="final",
                        source_iteration=None,
                    )
                )

    if not pass_frames:
        msg = f"No Palace eigenmode eig.csv files found under {output_dir}"
        raise FileNotFoundError(msg)

    return _add_eigenmode_convergence_columns(pd.concat(pass_frames, ignore_index=True))


def summarize_eigenmode_history(
    history: pd.DataFrame,
    *,
    include_imaginary: bool = False,
) -> pd.DataFrame:
    """Build per-pass summary rows from an Eigenmode history frame."""
    frame = _add_eigenmode_convergence_columns(history)
    if frame.empty:
        return frame

    summary = (
        frame.groupby(
            ["iteration_index", "label", "is_final"],
            sort=True,
            dropna=False,
        )
        .agg(
            n_modes=("mode_index", "nunique"),
            max_abs_delta_to_previous_mhz=("abs_delta_to_previous_mhz", "max"),
            max_abs_relative_delta_to_previous_percent=(
                "abs_relative_delta_to_previous_percent",
                "max",
            ),
            max_abs_delta_to_final_mhz=("abs_delta_to_final_mhz", "max"),
            max_abs_relative_delta_to_final_percent=(
                "abs_relative_delta_to_final_percent",
                "max",
            ),
            max_abs_imaginary_relative_delta_to_previous_percent=(
                "abs_imaginary_relative_delta_to_previous_percent",
                "max",
            ),
        )
        .reset_index()
    )
    if include_imaginary:
        summary["hfss_max_delta_freq_percent"] = summary[
            [
                "max_abs_relative_delta_to_previous_percent",
                "max_abs_imaginary_relative_delta_to_previous_percent",
            ]
        ].max(axis=1)
    else:
        summary["hfss_max_delta_freq_percent"] = summary[
            "max_abs_relative_delta_to_previous_percent"
        ]
    return summary.sort_values("iteration_index").reset_index(drop=True)


def load_eigenmode_report(
    source: str | Path | dict,
    *,
    index_map_path: str | Path | None = None,
    include_final: bool = True,
    require_epr: bool = False,
) -> EigenmodeReport:
    """Load final Eigenmode rows plus optional indexed EPR report tables.

    This is a thin composition layer over the stricter primitive loaders. The
    final ``eig.csv`` is required. Palace indexed reports are loaded
    independently when present and are reported as missing rather than forcing
    every Eigenmode run to emit all EPR families.
    """
    import pandas as pd

    eigenmodes = load_eigenmodes(source)
    source_rows: list[dict[str, Any]] = []
    source_rows.append(
        _report_source_row(
            "eig.csv",
            eigenmodes.source_path,
            required=True,
            present=True,
            loaded=True,
            message="loaded final eigenmode modes",
        )
    )

    mode_history = _load_eigenmode_history_for_report(
        source,
        eigenmodes,
        include_final=include_final,
        source_rows=source_rows,
    )
    pass_summary = (
        _empty_eigenmode_pass_summary()
        if mode_history.empty
        else summarize_eigenmode_history(mode_history)
    )

    resolved_index_map_path = _find_optional_postprocessing_index_map_path(
        source,
        index_map_path=index_map_path,
    )
    index_map_present = (
        resolved_index_map_path is not None and resolved_index_map_path.exists()
    )
    index_map_loaded = False
    if index_map_present:
        index_map = load_postprocessing_index_map(
            source,
            index_map_path=resolved_index_map_path,
        )
        index_map_frame = _postprocessing_index_map_to_dataframe(index_map)
        index_map_loaded = True
        index_message = "loaded postprocessing index map"
    else:
        index_map_frame = _empty_index_map_dataframe()
        index_message = "not found"
    source_rows.append(
        _report_source_row(
            "palace_index_map.json",
            resolved_index_map_path,
            required=False,
            present=index_map_present,
            loaded=index_map_loaded,
            message=index_message,
        )
    )

    resolved_config_path = _find_optional_config_path(source, config_path=None)
    config_present = resolved_config_path is not None and resolved_config_path.exists()
    config_loaded = False
    if config_present:
        domain_materials = load_domain_material_summary(
            source,
            config_path=resolved_config_path,
            index_map_path=resolved_index_map_path if index_map_present else None,
        )
        dielectric_interfaces = load_dielectric_interface_summary(
            source,
            config_path=resolved_config_path,
            index_map_path=resolved_index_map_path if index_map_present else None,
        )
        config_loaded = True
        config_message = (
            "loaded config material and interface summaries"
            if index_map_present
            else (
                "loaded config material and interface summaries without "
                "palace_index_map.json"
            )
        )
    else:
        domain_materials = _empty_domain_material_summary()
        dielectric_interfaces = _empty_dielectric_interface_summary()
        config_message = "not found"
    source_rows.append(
        _report_source_row(
            "config.json",
            resolved_config_path,
            required=False,
            present=config_present,
            loaded=config_loaded,
            message=config_message,
        )
    )

    domain_energy, domain_source = _load_optional_eigenmode_report_table(
        source,
        "domain-E.csv",
        loader=load_domain_energy_summary,
        empty_factory=_empty_domain_energy_summary,
        index_map_path=resolved_index_map_path,
        index_map_present=index_map_present,
    )
    source_rows.append(domain_source)
    surface_q, surface_source = _load_optional_eigenmode_report_table(
        source,
        "surface-Q.csv",
        loader=load_surface_q_summary,
        empty_factory=_empty_surface_q_summary,
        index_map_path=resolved_index_map_path,
        index_map_present=index_map_present,
    )
    source_rows.append(surface_source)
    port_epr, port_source = _load_optional_eigenmode_report_table(
        source,
        "port-EPR.csv",
        loader=load_port_epr_summary,
        empty_factory=_empty_port_epr_summary,
        index_map_path=resolved_index_map_path,
        index_map_present=index_map_present,
    )
    source_rows.append(port_source)

    if require_epr:
        required_failures = [
            row["name"]
            for row in (domain_source, surface_source)
            if not bool(row["loaded"])
        ]
        if required_failures:
            failure_names = ", ".join(str(name) for name in required_failures)
            msg = f"Missing required eigenmode EPR reports: {failure_names}"
            raise FileNotFoundError(msg)

    domain_loss = summarize_domain_loss(
        domain_energy,
        domain_materials,
        modes=eigenmodes,
    )
    surface_loss = summarize_surface_loss(
        surface_q,
        dielectric_interfaces,
        modes=eigenmodes,
    )
    surface_interface_summary = summarize_surface_q_by_interface(surface_loss)
    loss_budget = summarize_loss_budget(
        domain_loss,
        surface_loss,
        modes=eigenmodes,
    )
    return EigenmodeReport(
        eigenmodes=eigenmodes,
        mode_history=mode_history,
        pass_summary=pass_summary,
        domain_energy=domain_energy,
        domain_loss=domain_loss,
        surface_q=surface_q,
        surface_loss=surface_loss,
        surface_interface_summary=surface_interface_summary,
        loss_budget=loss_budget,
        port_epr=port_epr,
        index_map=index_map_frame,
        domain_materials=domain_materials,
        dielectric_interfaces=dielectric_interfaces,
        sources=pd.DataFrame.from_records(
            source_rows,
            columns=_REPORT_SOURCE_COLUMNS,
        ),
    )


def load_electrostatic_report(
    source: str | Path | dict,
    *,
    index_map_path: str | Path | None = None,
    terminal_names: tuple[str, ...] | list[str] | None = None,
    include_history: bool = True,
    require_epr: bool = False,
    frequency_ghz: float | None = None,
) -> ElectrostaticReport:
    """Load Electrostatic terminal matrices plus optional indexed EPR reports.

    Electrostatic Palace outputs do not carry a resonant frequency. Loss-rate
    and T1 columns are therefore derived only when ``frequency_ghz`` is passed
    explicitly; otherwise the report keeps inverse-Q and equivalent-Q columns.
    """
    import pandas as pd

    if frequency_ghz is not None:
        _validate_positive_frequency_ghz(frequency_ghz)

    source_rows: list[dict[str, Any]] = []
    resolved_index_map_path = _find_optional_postprocessing_index_map_path(
        source,
        index_map_path=index_map_path,
    )
    index_map_present = (
        resolved_index_map_path is not None and resolved_index_map_path.exists()
    )
    index_map_loaded = False
    if index_map_present:
        index_map = load_postprocessing_index_map(
            source,
            index_map_path=resolved_index_map_path,
        )
        index_map_frame = _postprocessing_index_map_to_dataframe(index_map)
        index_map_loaded = True
        index_message = "loaded postprocessing index map"
    else:
        index_map_frame = _empty_index_map_dataframe()
        index_message = "not found"
    source_rows.append(
        _report_source_row(
            "palace_index_map.json",
            resolved_index_map_path,
            required=False,
            present=index_map_present,
            loaded=index_map_loaded,
            message=index_message,
        )
    )

    terminal_c = _load_terminal_matrix_for_report(
        source,
        "C",
        index_map_path=resolved_index_map_path if index_map_present else None,
        terminal_names=terminal_names,
        source_rows=source_rows,
        required=True,
    )
    if terminal_c is None:
        msg = "Required electrostatic terminal-C.csv not found"
        raise FileNotFoundError(msg)
    terminal_cm = _load_terminal_matrix_for_report(
        source,
        "Cm",
        index_map_path=resolved_index_map_path if index_map_present else None,
        terminal_names=terminal_names,
        source_rows=source_rows,
        required=False,
    )
    terminal_cinv = _load_terminal_matrix_for_report(
        source,
        "Cinv",
        index_map_path=resolved_index_map_path if index_map_present else None,
        terminal_names=terminal_names,
        source_rows=source_rows,
        required=False,
    )

    terminal_c_history = _load_terminal_matrix_history_for_report(
        source,
        terminal_c,
        include_history=include_history,
        index_map_path=resolved_index_map_path if index_map_present else None,
        terminal_names=terminal_names,
        source_rows=source_rows,
    )
    terminal_c_pass_summary = (
        _empty_terminal_matrix_pass_summary()
        if terminal_c_history.empty
        else summarize_terminal_matrix_history(terminal_c_history)
    )
    terminal_cm_history = _load_terminal_matrix_history_for_report(
        source,
        terminal_cm,
        include_history=include_history,
        index_map_path=resolved_index_map_path if index_map_present else None,
        terminal_names=terminal_names,
        source_rows=source_rows,
    )
    terminal_cm_pass_summary = (
        _empty_terminal_matrix_pass_summary()
        if terminal_cm_history.empty
        else summarize_terminal_matrix_history(terminal_cm_history)
    )
    terminal_cinv_history = _load_terminal_matrix_history_for_report(
        source,
        terminal_cinv,
        include_history=include_history,
        index_map_path=resolved_index_map_path if index_map_present else None,
        terminal_names=terminal_names,
        source_rows=source_rows,
    )
    terminal_cinv_pass_summary = (
        _empty_terminal_matrix_pass_summary()
        if terminal_cinv_history.empty
        else summarize_terminal_matrix_history(terminal_cinv_history)
    )

    resolved_config_path = _find_optional_config_path(source, config_path=None)
    config_present = resolved_config_path is not None and resolved_config_path.exists()
    config_loaded = False
    if config_present:
        domain_materials = load_domain_material_summary(
            source,
            config_path=resolved_config_path,
            index_map_path=resolved_index_map_path if index_map_present else None,
        )
        dielectric_interfaces = load_dielectric_interface_summary(
            source,
            config_path=resolved_config_path,
            index_map_path=resolved_index_map_path if index_map_present else None,
        )
        config_loaded = True
        config_message = (
            "loaded config material and interface summaries"
            if index_map_present
            else (
                "loaded config material and interface summaries without "
                "palace_index_map.json"
            )
        )
    else:
        domain_materials = _empty_domain_material_summary()
        dielectric_interfaces = _empty_dielectric_interface_summary()
        config_message = "not found"
    source_rows.append(
        _report_source_row(
            "config.json",
            resolved_config_path,
            required=False,
            present=config_present,
            loaded=config_loaded,
            message=config_message,
        )
    )

    domain_energy, domain_source = _load_optional_eigenmode_report_table(
        source,
        "domain-E.csv",
        loader=load_domain_energy_summary,
        empty_factory=_empty_domain_energy_summary,
        index_map_path=resolved_index_map_path,
        index_map_present=index_map_present,
    )
    source_rows.append(domain_source)
    surface_q, surface_source = _load_optional_eigenmode_report_table(
        source,
        "surface-Q.csv",
        loader=load_surface_q_summary,
        empty_factory=_empty_surface_q_summary,
        index_map_path=resolved_index_map_path,
        index_map_present=index_map_present,
    )
    source_rows.append(surface_source)

    if require_epr:
        required_failures = [
            row["name"]
            for row in (domain_source, surface_source)
            if not bool(row["loaded"])
        ]
        if required_failures:
            failure_names = ", ".join(str(name) for name in required_failures)
            msg = f"Missing required electrostatic EPR reports: {failure_names}"
            raise FileNotFoundError(msg)

    domain_loss = summarize_domain_loss(
        domain_energy,
        domain_materials,
        frequency_ghz=frequency_ghz,
    )
    surface_loss = summarize_surface_loss(
        surface_q,
        dielectric_interfaces,
        frequency_ghz=frequency_ghz,
    )
    surface_interface_summary = summarize_surface_q_by_interface(surface_loss)
    loss_budget = _summarize_electrostatic_loss_budget(
        domain_loss,
        surface_loss,
        frequency_ghz=frequency_ghz,
    )
    return ElectrostaticReport(
        terminal_c=terminal_c,
        terminal_cm=terminal_cm,
        terminal_cinv=terminal_cinv,
        terminal_c_history=terminal_c_history,
        terminal_cm_history=terminal_cm_history,
        terminal_cinv_history=terminal_cinv_history,
        terminal_c_pass_summary=terminal_c_pass_summary,
        terminal_cm_pass_summary=terminal_cm_pass_summary,
        terminal_cinv_pass_summary=terminal_cinv_pass_summary,
        domain_materials=domain_materials,
        dielectric_interfaces=dielectric_interfaces,
        domain_energy=domain_energy,
        domain_loss=domain_loss,
        surface_q=surface_q,
        surface_loss=surface_loss,
        surface_interface_summary=surface_interface_summary,
        loss_budget=loss_budget,
        index_map=index_map_frame,
        sources=pd.DataFrame.from_records(
            source_rows,
            columns=_REPORT_SOURCE_COLUMNS,
        ),
    )


def load_postprocessing_index_map(
    source: str | Path | dict,
    *,
    index_map_path: str | Path | None = None,
) -> PostprocessingIndexMap:
    """Load ``palace_index_map.json`` as a postprocessing index map.

    Args:
        source: Simulation directory, Palace output directory, results dict, or
            any path near the generated index-map artifact.
        index_map_path: Optional explicit ``palace_index_map.json`` path.

    Returns:
        :class:`gsim.palace.mesh.PostprocessingIndexMap`.
    """
    from gsim.palace.mesh.postprocessing import (
        PostprocessingIndexEntry,
        PostprocessingIndexMap,
    )

    path = (
        Path(index_map_path)
        if index_map_path is not None
        else _find_postprocessing_index_map(source)
    )
    if path is None or not path.exists():
        msg = "palace_index_map.json not found"
        raise FileNotFoundError(msg)

    data = json.loads(path.read_text())
    entries = tuple(
        PostprocessingIndexEntry(
            section=str(row["section"]),
            index=int(row["index"]),
            entry_name=str(row.get("entry_name", row.get("physical_name", ""))),
            role=str(row.get("role", "")),
            attributes=tuple(int(v) for v in row.get("attributes", ())),
            entity_tags=tuple(int(v) for v in row.get("entity_tags", ())),
            physical_names=tuple(str(v) for v in row.get("physical_names", ())),
            dimension=_optional_int(row.get("dimension")),
            source=_optional_str(row.get("source")),
            interface_of=_optional_str_pair(row.get("interface_of")),
            exterior_of=_optional_str(row.get("exterior_of")),
            metadata=_as_mapping(row.get("metadata")),
            extra={
                key: value
                for key, value in row.items()
                if key
                not in {
                    "section",
                    "index",
                    "entry_name",
                    "role",
                    "attributes",
                    "entity_tags",
                    "physical_names",
                    "dimension",
                    "source",
                    "interface_of",
                    "exterior_of",
                    "metadata",
                }
            },
        )
        for row in data.get("entries", ())
    )
    return PostprocessingIndexMap(
        entries=entries,
        schema_version=int(data.get("schema_version", 1)),
    )


def load_terminal_matrix(
    source: str | Path | dict,
    matrix_kind: str = "C",
    *,
    index_map_path: str | Path | None = None,
    terminal_names: tuple[str, ...] | list[str] | None = None,
    display_scale: float | None = None,
    display_unit: str | None = None,
) -> TerminalMatrix:
    """Load a Palace electrostatic terminal matrix with named terminals.

    Args:
        source: CSV path, simulation directory, Palace output directory, or
            results dict.
        matrix_kind: One of ``"C"``, ``"Cm"``, or ``"Cinv"``.
        index_map_path: Optional explicit ``palace_index_map.json`` path.
        terminal_names: Optional explicit terminal labels. If omitted, labels
            come from ``Boundaries.Terminal`` rows in ``palace_index_map.json``.
        display_scale: Optional display scale. Defaults to fF for capacitance
            matrices and 1 for inverse capacitance.
        display_unit: Optional display unit label.

    Returns:
        :class:`TerminalMatrix` with SI values and display helpers.
    """
    kind = _normalize_terminal_matrix_kind(matrix_kind)
    csv_path = _resolve_terminal_matrix_csv(source, kind)
    return _load_terminal_matrix_from_csv(
        csv_path,
        kind,
        label_source=source,
        index_map_path=index_map_path,
        terminal_names=terminal_names,
        display_scale=display_scale,
        display_unit=display_unit,
    )


def load_terminal_matrix_history(
    source: str | Path,
    matrix_kind: str = "C",
    *,
    index_map_path: str | Path | None = None,
    terminal_names: tuple[str, ...] | list[str] | None = None,
    include_final: bool = True,
    display_scale: float | None = None,
    display_unit: str | None = None,
) -> pd.DataFrame:
    """Load Palace electrostatic terminal matrices across AMR passes.

    ``source`` should be the Palace output directory or simulation directory.
    Iteration directories named ``iterationN`` or ``iterationNN`` are treated as
    AMR passes; the final matrix is appended unless it duplicates the last pass.
    """
    import pandas as pd

    base = Path(source)
    if base.is_file():
        msg = "load_terminal_matrix_history() expects a simulation/output directory."
        raise ValueError(msg)

    kind = _normalize_terminal_matrix_kind(matrix_kind)
    pass_frames: list[pd.DataFrame] = []
    last_raw: pd.DataFrame | None = None

    for iteration_dir, pass_index in _iteration_dirs(base):
        csv_path = iteration_dir / str(_TERMINAL_MATRIX_SPECS[kind]["file_name"])
        if not csv_path.exists():
            continue
        matrix = _load_terminal_matrix_from_csv(
            csv_path,
            kind,
            label_source=base,
            index_map_path=index_map_path,
            terminal_names=terminal_names,
            display_scale=display_scale,
            display_unit=display_unit,
        )
        pass_frames.append(
            _terminal_matrix_to_history_frame(
                matrix,
                pass_index=pass_index,
                label=f"Pass {pass_index}",
                is_final=False,
            )
        )
        last_raw = _read_terminal_matrix_csv(csv_path)

    if include_final:
        final_csv_path = _find_terminal_matrix_final_csv(base, kind)
        if final_csv_path is not None:
            final_raw = _read_terminal_matrix_csv(final_csv_path)
            if last_raw is None or not _terminal_matrices_match(last_raw, final_raw):
                matrix = _load_terminal_matrix_from_csv(
                    final_csv_path,
                    kind,
                    label_source=base,
                    index_map_path=index_map_path,
                    terminal_names=terminal_names,
                    display_scale=display_scale,
                    display_unit=display_unit,
                )
                pass_index = 1 if not pass_frames else _next_pass_index(pass_frames)
                pass_frames.append(
                    _terminal_matrix_to_history_frame(
                        matrix,
                        pass_index=pass_index,
                        label="Final",
                        is_final=True,
                    )
                )

    if not pass_frames:
        csv_name = _TERMINAL_MATRIX_SPECS[kind]["file_name"]
        msg = f"No Palace electrostatic {csv_name} files found under {base}"
        raise FileNotFoundError(msg)

    return _add_terminal_matrix_convergence_columns(
        pd.concat(pass_frames, ignore_index=True)
    )


def summarize_terminal_matrix_history(history: pd.DataFrame) -> pd.DataFrame:
    """Build per-pass summary rows from terminal matrix history."""
    frame = _add_terminal_matrix_convergence_columns(history)
    if frame.empty:
        return frame

    summary = (
        frame.groupby(
            ["matrix_kind", "pass_index", "label", "is_final", "display_unit"],
            sort=True,
        )
        .agg(
            n_elements=("element", "nunique"),
            n_diagonal_elements=("is_diagonal", "sum"),
            max_abs_value=("value_si", lambda column: column.abs().max()),
            max_abs_display_value=("display_value", lambda column: column.abs().max()),
            max_abs_delta_to_previous=("abs_delta_to_previous_si", "max"),
            max_abs_display_delta_to_previous=(
                "abs_display_delta_to_previous",
                "max",
            ),
            max_abs_relative_delta_to_previous_percent=(
                "abs_relative_delta_to_previous_percent",
                "max",
            ),
            max_abs_delta_to_final=("abs_delta_to_final_si", "max"),
            max_abs_display_delta_to_final=("abs_display_delta_to_final", "max"),
            max_abs_relative_delta_to_final_percent=(
                "abs_relative_delta_to_final_percent",
                "max",
            ),
        )
        .reset_index()
    )
    summary["n_off_diagonal_elements"] = (
        summary["n_elements"] - summary["n_diagonal_elements"]
    )
    return summary.sort_values(["matrix_kind", "pass_index"]).reset_index(drop=True)


def load_indexed_csv(
    source: str | Path | dict,
    csv_name: str | None = None,
    *,
    section: str | None = None,
    index_map_path: str | Path | None = None,
    rename_columns: bool = True,
) -> IndexedCsv:
    """Load a Palace indexed CSV and annotate columns from the index map.

    Palace postprocessing reports use columns such as ``E_elec[1] (J)`` and
    ``p_surf[2]``. This loader preserves the numeric report contract while
    linking each index back to the generated ``palace_index_map.json`` artifact.

    Args:
        source: CSV path, simulation directory, Palace output directory, or
            results dict.
        csv_name: CSV name when ``source`` is a directory or results dict.
        section: Palace index-map section. If omitted, common Palace report
            filenames infer the section.
        index_map_path: Optional explicit ``palace_index_map.json`` path.
        rename_columns: If true, indexed columns with known physical names are
            renamed from ``quantity[1]`` to ``quantity[physical-name]``.

    Returns:
        :class:`IndexedCsv` containing the DataFrame and column provenance rows.
    """
    import pandas as pd

    csv_path = _resolve_report_csv(source, csv_name)
    resolved_section = section or _section_for_csv(csv_path.name)
    index_map = load_postprocessing_index_map(source, index_map_path=index_map_path)
    frame = pd.read_csv(csv_path)
    frame.columns = frame.columns.str.strip()

    rename_map: dict[str, str] = {}
    column_rows: list[IndexedCsvColumn] = []
    for column in frame.columns:
        parsed = _parse_indexed_csv_column(column)
        if parsed is None:
            continue
        quantity, index, suffix = parsed
        entry = index_map.entry_for_index(resolved_section, index)
        physical_name = None if entry is None else entry.primary_physical_name
        renamed = (
            f"{quantity}[{physical_name}]{suffix}"
            if rename_columns and physical_name is not None
            else column
        )
        if renamed != column:
            rename_map[column] = renamed
        column_rows.append(
            IndexedCsvColumn(
                original_name=column,
                renamed_name=renamed,
                quantity=quantity,
                index=index,
                section=resolved_section,
                physical_name=physical_name,
                entry_name=None if entry is None else entry.entry_name,
                role=None if entry is None else entry.role,
                attributes=() if entry is None else entry.attributes,
                metadata={} if entry is None else dict(entry.metadata),
                extra={} if entry is None else dict(entry.extra),
            )
        )

    if rename_map:
        frame = frame.rename(columns=rename_map)
    return IndexedCsv(
        source_path=csv_path,
        section=resolved_section,
        dataframe=frame,
        columns=tuple(column_rows),
    )


def load_domain_energy_summary(
    source: str | Path | dict,
    *,
    index_map_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load ``domain-E.csv`` as one row per sampled domain index.

    The returned frame preserves Palace quantities such as ``E_elec`` and
    ``p_elec`` while attaching physical-name provenance from
    ``palace_index_map.json``.
    """
    return _load_indexed_quantity_summary(
        source,
        "domain-E.csv",
        index_name="domain_index",
        quantity_columns={
            "E_elec": "E_elec_j",
            "E_mag": "E_mag_j",
            "p_elec": "p_elec",
            "p_mag": "p_mag",
        },
        index_map_path=index_map_path,
    )


def load_surface_q_summary(
    source: str | Path | dict,
    *,
    index_map_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load ``surface-Q.csv`` as one row per sampled surface index."""
    frame = _load_indexed_quantity_summary(
        source,
        "surface-Q.csv",
        index_name="surface_index",
        quantity_columns={
            "p_surf": "p_surf",
            "Q_surf": "q_surf",
        },
        index_map_path=index_map_path,
    )
    if "q_surf" in frame.columns:
        frame["inverse_q"] = _inverse_q_values(frame["q_surf"])
    return frame


def summarize_surface_q_by_interface(
    surface_summary: pd.DataFrame,
    *,
    include_interface_types: tuple[str, ...] = ("MA", "MS", "SA"),
) -> pd.DataFrame:
    """Summarize a surface-Q frame by interface type."""
    import pandas as pd

    columns = [
        "interface_type",
        "surface_count",
        "p_surf_sum",
        "inverse_q_sum",
        "q_equivalent",
        "p_surf_fraction",
        "inverse_q_fraction",
    ]
    if surface_summary.empty:
        return pd.DataFrame(columns=columns)

    frame = surface_summary.copy()
    if "interface_type" not in frame.columns:
        frame["interface_type"] = ""
    if "p_surf" not in frame.columns:
        frame["p_surf"] = 0.0
    if "inverse_q" not in frame.columns:
        frame["inverse_q"] = 0.0

    interface_order = list(include_interface_types)
    for interface_type in frame["interface_type"].dropna().astype(str).unique():
        if interface_type and interface_type not in interface_order:
            interface_order.append(interface_type)

    rows: list[dict[str, Any]] = []
    for interface_type in interface_order:
        subset = frame.loc[frame["interface_type"] == interface_type]
        p_surf_sum = float(subset["p_surf"].sum()) if not subset.empty else 0.0
        inverse_q_sum = float(subset["inverse_q"].sum()) if not subset.empty else 0.0
        rows.append(
            {
                "interface_type": interface_type,
                "surface_count": len(subset),
                "p_surf_sum": p_surf_sum,
                "inverse_q_sum": inverse_q_sum,
                "q_equivalent": _q_from_inverse_q(inverse_q_sum),
            }
        )

    summary = pd.DataFrame.from_records(rows, columns=columns[:5])
    p_total = _sum_numeric_column(summary, "p_surf_sum")
    inverse_q_total = _sum_numeric_column(summary, "inverse_q_sum")
    summary["p_surf_fraction"] = [
        _fraction(float(value), p_total) for value in summary["p_surf_sum"]
    ]
    summary["inverse_q_fraction"] = [
        _fraction(float(value), inverse_q_total) for value in summary["inverse_q_sum"]
    ]
    return cast("pd.DataFrame", summary.loc[:, columns])


def summarize_domain_loss(
    domain_energy: pd.DataFrame,
    domain_materials: pd.DataFrame,
    *,
    modes: Eigenmodes | pd.DataFrame | None = None,
    frequency_ghz: float | None = None,
) -> pd.DataFrame:
    """Combine domain EPR rows with effective material loss parameters.

    ``domain-E.csv`` contains participation values, while ``config.json``
    contains the effective Palace material loss tangent. This helper keeps
    those primitive loaders separate and derives ``inverse_q = p_elec *
    loss_tangent`` only in this report layer.
    """
    import pandas as pd

    if domain_energy.empty:
        return _empty_domain_loss_summary()

    rows: list[dict[str, Any]] = []
    for _, domain_row in domain_energy.iterrows():
        row = dict(domain_row)
        material_row = _matching_domain_material_row(domain_row, domain_materials)
        loss_tangent = _numeric_or_default(
            None if material_row is None else material_row.get("loss_tangent"),
            default=0.0,
        )
        p_elec = _numeric_or_default(domain_row.get("p_elec"), default=0.0)
        inverse_q = p_elec * loss_tangent
        mode_index = _optional_int(domain_row.get("mode_index"))
        resolved_frequency = _frequency_for_mode(
            mode_index,
            modes=modes,
            frequency_ghz=frequency_ghz,
        )

        row.update(
            {
                "frequency_ghz": resolved_frequency,
                "material_attribute": (
                    None
                    if material_row is None
                    else material_row.get("material_attribute")
                ),
                "material_attributes": (
                    ()
                    if material_row is None
                    else material_row.get("material_attributes", ())
                ),
                "material_name": (
                    None if material_row is None else material_row.get("material_name")
                ),
                "material_permittivity": (
                    None if material_row is None else material_row.get("permittivity")
                ),
                "material_loss_tangent": loss_tangent,
                "material_conductivity": (
                    None if material_row is None else material_row.get("conductivity")
                ),
                "material_permeability": (
                    None if material_row is None else material_row.get("permeability")
                ),
                "material_axes": (
                    None if material_row is None else material_row.get("material_axes")
                ),
                "loss_tangent": loss_tangent,
                "inverse_q": inverse_q,
                "q_equivalent": _q_from_inverse_q(inverse_q),
                **_rate_columns_for_frequency(
                    frequency_ghz=resolved_frequency,
                    inverse_q=inverse_q,
                ),
            }
        )
        rows.append(row)

    if not rows:
        return _empty_domain_loss_summary()
    return _ordered_dataframe(pd.DataFrame.from_records(rows), _DOMAIN_LOSS_COLUMNS)


def summarize_surface_loss(
    surface_q: pd.DataFrame,
    dielectric_interfaces: pd.DataFrame | None = None,
    *,
    modes: Eigenmodes | pd.DataFrame | None = None,
    frequency_ghz: float | None = None,
) -> pd.DataFrame:
    """Combine surface-Q rows with configured dielectric interface parameters.

    Palace already reports the effective surface ``Q_surf``. This helper keeps
    that solver result authoritative, adds configured interface metadata, and
    derives rate/T1 columns only when mode frequency is available.
    """
    import pandas as pd

    if surface_q.empty:
        return _empty_surface_loss_summary()

    interface_frame = (
        dielectric_interfaces
        if dielectric_interfaces is not None
        else _empty_dielectric_interface_summary()
    )
    rows: list[dict[str, Any]] = []
    for _, surface_row in surface_q.iterrows():
        row = dict(surface_row)
        interface_row = _matching_dielectric_interface_row(
            surface_row,
            interface_frame,
        )
        inverse_q = _numeric_or_default(surface_row.get("inverse_q"), default=0.0)
        mode_index = _optional_int(surface_row.get("mode_index"))
        resolved_frequency = _frequency_for_mode(
            mode_index,
            modes=modes,
            frequency_ghz=frequency_ghz,
        )
        row.update(
            {
                "frequency_ghz": resolved_frequency,
                "q_equivalent": _q_from_inverse_q(inverse_q),
                "surface_attribute": (
                    None
                    if interface_row is None
                    else interface_row.get("surface_attribute")
                ),
                "surface_attributes": (
                    ()
                    if interface_row is None
                    else interface_row.get("surface_attributes", ())
                ),
                "preset_name": (
                    None if interface_row is None else interface_row.get("preset_name")
                ),
                "preset_source": (
                    None
                    if interface_row is None
                    else interface_row.get("preset_source")
                ),
                "thickness": None
                if interface_row is None
                else interface_row.get("thickness"),
                "permittivity": (
                    None if interface_row is None else interface_row.get("permittivity")
                ),
                "loss_tangent": (
                    None if interface_row is None else interface_row.get("loss_tangent")
                ),
                **_rate_columns_for_frequency(
                    frequency_ghz=resolved_frequency,
                    inverse_q=inverse_q,
                ),
            }
        )
        rows.append(row)

    if not rows:
        return _empty_surface_loss_summary()
    return _ordered_dataframe(pd.DataFrame.from_records(rows), _SURFACE_LOSS_COLUMNS)


def summarize_loss_budget(
    domain_loss: pd.DataFrame,
    surface_loss: pd.DataFrame,
    *,
    modes: Eigenmodes | pd.DataFrame | None = None,
    frequency_ghz: float | None = None,
) -> pd.DataFrame:
    """Summarize per-mode bulk/domain and surface inverse-Q contributions."""
    import pandas as pd

    if domain_loss.empty and surface_loss.empty:
        return _empty_loss_budget_summary()

    mode_indices = _loss_mode_indices(domain_loss, surface_loss, modes)
    if not mode_indices and frequency_ghz is not None:
        mode_indices = (None,)
    if not mode_indices:
        return _empty_loss_budget_summary()

    rows: list[dict[str, Any]] = []
    for mode_index in mode_indices:
        domain_rows = _rows_for_mode(domain_loss, mode_index)
        surface_rows = _rows_for_mode(surface_loss, mode_index)
        domain_inverse_q = _sum_numeric_column(domain_rows, "inverse_q")
        surface_inverse_q = _sum_numeric_column(surface_rows, "inverse_q")
        total_inverse_q = domain_inverse_q + surface_inverse_q
        q_eig = _q_for_mode(mode_index, modes=modes)
        inverse_q_eig = _inverse_q_from_q(q_eig)
        eig_with_surface_inverse_q = inverse_q_eig + surface_inverse_q
        resolved_frequency = _frequency_for_mode(
            mode_index,
            modes=modes,
            frequency_ghz=frequency_ghz,
        )
        rows.append(
            {
                "mode_index": mode_index,
                "frequency_ghz": resolved_frequency,
                "q_eig": q_eig,
                "inverse_q_eig": inverse_q_eig,
                "domain_inverse_q_sum": domain_inverse_q,
                "surface_inverse_q_sum": surface_inverse_q,
                "total_inverse_q_sum": total_inverse_q,
                "eig_with_surface_inverse_q_sum": eig_with_surface_inverse_q,
                "q_total": _q_from_inverse_q(total_inverse_q),
                "q_eig_with_surface": _q_from_inverse_q(eig_with_surface_inverse_q),
                "domain_vs_eig_relative_error": _relative_error(
                    domain_inverse_q,
                    inverse_q_eig,
                ),
                **_rate_columns_for_frequency(
                    frequency_ghz=resolved_frequency,
                    inverse_q=total_inverse_q,
                ),
            }
        )

    return _ordered_dataframe(pd.DataFrame.from_records(rows), _LOSS_BUDGET_COLUMNS)


def _summarize_electrostatic_loss_budget(
    domain_loss: pd.DataFrame,
    surface_loss: pd.DataFrame,
    *,
    frequency_ghz: float | None = None,
) -> pd.DataFrame:
    """Build an Electrostatic aggregate when Palace rows have no mode index."""
    import pandas as pd

    group_columns = _electrostatic_loss_group_columns(domain_loss, surface_loss)
    if group_columns:
        group_values = _electrostatic_loss_group_values(
            domain_loss,
            surface_loss,
            group_columns,
        )
        rows = [
            _electrostatic_loss_budget_row(
                _rows_for_electrostatic_loss_group(domain_loss, values),
                _rows_for_electrostatic_loss_group(surface_loss, values),
                values,
                frequency_ghz=frequency_ghz,
            )
            for values in group_values
        ]
        return _ordered_dataframe(
            pd.DataFrame.from_records(rows),
            _LOSS_BUDGET_COLUMNS,
        )

    budget = summarize_loss_budget(
        domain_loss,
        surface_loss,
        frequency_ghz=frequency_ghz,
    )
    if not budget.empty or (domain_loss.empty and surface_loss.empty):
        return budget

    rows = [
        _electrostatic_loss_budget_row(
            domain_loss,
            surface_loss,
            {},
            frequency_ghz=frequency_ghz,
        )
    ]
    return _ordered_dataframe(pd.DataFrame.from_records(rows), _LOSS_BUDGET_COLUMNS)


def _electrostatic_loss_budget_row(
    domain_loss: pd.DataFrame,
    surface_loss: pd.DataFrame,
    group_values: dict[str, Any],
    *,
    frequency_ghz: float | None,
) -> dict[str, Any]:
    domain_inverse_q = _sum_numeric_column(domain_loss, "inverse_q")
    surface_inverse_q = _sum_numeric_column(surface_loss, "inverse_q")
    total_inverse_q = domain_inverse_q + surface_inverse_q
    return {
        **group_values,
        "mode_index": None,
        "frequency_ghz": frequency_ghz,
        "q_eig": float("nan"),
        "inverse_q_eig": float("nan"),
        "domain_inverse_q_sum": domain_inverse_q,
        "surface_inverse_q_sum": surface_inverse_q,
        "total_inverse_q_sum": total_inverse_q,
        "eig_with_surface_inverse_q_sum": float("nan"),
        "q_total": _q_from_inverse_q(total_inverse_q),
        "q_eig_with_surface": float("nan"),
        "domain_vs_eig_relative_error": float("nan"),
        **_rate_columns_for_frequency(
            frequency_ghz=frequency_ghz,
            inverse_q=total_inverse_q,
        ),
    }


def _electrostatic_loss_group_columns(
    domain_loss: pd.DataFrame,
    surface_loss: pd.DataFrame,
) -> tuple[str, ...]:
    frames = [frame for frame in (domain_loss, surface_loss) if not frame.empty]
    if not frames:
        return ()
    common_columns = set(frames[0].columns)
    for frame in frames[1:]:
        common_columns &= set(frame.columns)

    if "source_index" in common_columns:
        columns = ["source_index"]
        columns.extend(
            column
            for column in ("sample_column", "sample_value")
            if column in common_columns
        )
        return tuple(columns)

    if {"sample_column", "sample_value"} <= common_columns:
        return ("sample_column", "sample_value")
    return ()


def _electrostatic_loss_group_values(
    domain_loss: pd.DataFrame,
    surface_loss: pd.DataFrame,
    group_columns: tuple[str, ...],
) -> list[dict[str, Any]]:
    if not group_columns:
        return []

    import pandas as pd

    frames = [
        frame.loc[:, list(group_columns)]
        for frame in (domain_loss, surface_loss)
        if not frame.empty
    ]
    if not frames:
        return []
    groups = pd.concat(frames, ignore_index=True).drop_duplicates()
    groups = groups.sort_values(list(group_columns)).reset_index(drop=True)
    return [dict(row) for row in groups.to_dict(orient="records")]


def _rows_for_electrostatic_loss_group(
    frame: pd.DataFrame,
    group_values: dict[str, Any],
) -> pd.DataFrame:
    if frame.empty or not group_values:
        return frame

    import pandas as pd

    mask = pd.Series(True, index=frame.index)
    for column, value in group_values.items():
        if column not in frame.columns:
            continue
        if pd.isna(value):
            mask &= frame[column].isna()
        else:
            mask &= frame[column] == value
    return cast("pd.DataFrame", frame.loc[mask])


def load_port_epr_summary(
    source: str | Path | dict,
    *,
    index_map_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load ``port-EPR.csv`` as one row per sampled port/surface-flux index."""
    frame = _load_indexed_quantity_summary(
        source,
        "port-EPR.csv",
        index_name="port_index",
        quantity_columns={
            "p": "p_port",
            "p_port": "p_port",
        },
        index_map_path=index_map_path,
    )
    if "p_port" in frame.columns:
        frame["abs_p_port"] = frame["p_port"].abs()
        group_columns = _sample_group_columns(frame)
        if group_columns:
            totals = frame.groupby(group_columns, dropna=False)["abs_p_port"].transform(
                "sum"
            )
        else:
            totals = frame["abs_p_port"].sum()
        frame["abs_p_port_fraction"] = _fraction_values(frame["abs_p_port"], totals)
    return frame


def load_domain_material_summary(
    source: str | Path | dict,
    *,
    config_path: str | Path | None = None,
    index_map_path: str | Path | None = None,
    material_resolution_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load Palace ``Domains.Materials`` with physical-name provenance.

    The returned table interprets the effective material rows from
    ``config.json`` and, when available, joins each material attribute back to
    the domain postprocessing entries in ``palace_index_map.json``. When
    available, ``palace_material_resolution.json`` is joined by material
    attribute so reports can explain which material overlay/model source was
    applied during config generation.
    """
    import pandas as pd

    resolved_config_path = _find_optional_config_path(
        source,
        config_path=config_path,
    )
    if resolved_config_path is None or not resolved_config_path.exists():
        msg = "config.json not found"
        raise FileNotFoundError(msg)

    data = json.loads(resolved_config_path.read_text())
    materials = _domain_material_entries(data)
    index_map = _load_optional_domain_material_index_map(
        source,
        index_map_path=index_map_path,
    )
    material_resolution_rows = _load_optional_material_resolution_rows(
        source,
        material_resolution_path=material_resolution_path,
    )

    rows: list[dict[str, Any]] = []
    for material_row_index, material in enumerate(materials, start=1):
        attributes = _material_attributes(material)
        if not attributes:
            rows.append(
                _domain_material_row(
                    material_row_index=material_row_index,
                    material_attribute=None,
                    material_attributes=attributes,
                    material=material,
                    index_entry=None,
                    material_resolution=_matching_material_resolution_row(
                        material_resolution_rows,
                        material_row_index=material_row_index,
                        material_attribute=None,
                    ),
                )
            )
            continue

        for attribute in attributes:
            material_resolution = _matching_material_resolution_row(
                material_resolution_rows,
                material_row_index=material_row_index,
                material_attribute=attribute,
            )
            matches = (
                index_map.entries_for_attribute(
                    attribute,
                    section="Domains.Postprocessing.Energy",
                )
                if index_map is not None
                else ()
            )
            if not matches:
                rows.append(
                    _domain_material_row(
                        material_row_index=material_row_index,
                        material_attribute=attribute,
                        material_attributes=attributes,
                        material=material,
                        index_entry=None,
                        material_resolution=material_resolution,
                    )
                )
                continue
            rows.extend(
                _domain_material_row(
                    material_row_index=material_row_index,
                    material_attribute=attribute,
                    material_attributes=attributes,
                    material=material,
                    index_entry=index_entry,
                    material_resolution=material_resolution,
                )
                for index_entry in matches
            )

    if not rows:
        return _empty_domain_material_summary()
    return pd.DataFrame.from_records(rows, columns=_DOMAIN_MATERIAL_COLUMNS)


def load_dielectric_interface_summary(
    source: str | Path | dict,
    *,
    config_path: str | Path | None = None,
    index_map_path: str | Path | None = None,
    material_resolution_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load Palace dielectric postprocessing interfaces with provenance.

    The returned table interprets ``Boundaries.Postprocessing.Dielectric`` rows
    from ``config.json`` and joins their Palace indices back to
    ``palace_index_map.json`` physical names when that map is available.
    """
    import pandas as pd

    resolved_config_path = _find_optional_config_path(
        source,
        config_path=config_path,
    )
    if resolved_config_path is None or not resolved_config_path.exists():
        msg = "config.json not found"
        raise FileNotFoundError(msg)

    data = json.loads(resolved_config_path.read_text())
    interfaces = _dielectric_interface_entries(data)
    index_map = _load_optional_postprocessing_index_map(
        source,
        index_map_path=index_map_path,
    )
    material_resolution_rows = _load_optional_interface_material_resolution_rows(
        source,
        material_resolution_path=material_resolution_path,
    )

    rows: list[dict[str, Any]] = []
    for interface_row_index, interface in enumerate(interfaces, start=1):
        surface_index = _optional_int(_config_material_value(interface, "Index"))
        attributes = _material_attributes(interface)
        interface_resolution = _matching_interface_material_resolution_row(
            material_resolution_rows,
            interface_row_index=interface_row_index,
            surface_index=surface_index,
        )
        matches = _dielectric_interface_matches(
            index_map,
            surface_index=surface_index,
            attributes=attributes,
        )
        if not matches:
            rows.append(
                _dielectric_interface_row(
                    interface_row_index=interface_row_index,
                    surface_index=surface_index,
                    surface_attribute=attributes[0] if attributes else None,
                    surface_attributes=attributes,
                    interface=interface,
                    index_entry=None,
                    material_resolution=interface_resolution,
                )
            )
            continue
        rows.extend(
            _dielectric_interface_row(
                interface_row_index=interface_row_index,
                surface_index=surface_index,
                surface_attribute=_matching_surface_attribute(index_entry, attributes),
                surface_attributes=attributes,
                interface=interface,
                index_entry=index_entry,
                material_resolution=interface_resolution,
            )
            for index_entry in matches
        )

    if not rows:
        return _empty_dielectric_interface_summary()
    return pd.DataFrame.from_records(rows, columns=_DIELECTRIC_INTERFACE_COLUMNS)


def get_port_map(source: str | Path | dict) -> dict[int, str]:
    """Return the ``{port_number: port_name}`` mapping.

    Accepts a directory path or a results dict from ``sim.run()``.
    """
    csv_path, base_dir = _resolve_source(source, require_csv=False)
    return _load_port_map(base_dir, csv_path)


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------


def _report_source_row(
    name: str,
    path: str | Path | None,
    *,
    required: bool,
    present: bool,
    loaded: bool,
    message: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "path": None if path is None else str(path),
        "required": required,
        "present": present,
        "loaded": loaded,
        "message": message,
    }


def _domain_material_entries(data: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(data, dict):
        return ()
    domains = data.get("Domains", {})
    if not isinstance(domains, dict):
        return ()
    materials = domains.get("Materials", ())
    if not isinstance(materials, (list, tuple)):
        return ()
    return tuple(entry for entry in materials if isinstance(entry, dict))


def _dielectric_interface_entries(data: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(data, dict):
        return ()
    boundaries = data.get("Boundaries", {})
    if not isinstance(boundaries, dict):
        return ()
    postprocessing = boundaries.get("Postprocessing", {})
    if not isinstance(postprocessing, dict):
        return ()
    interfaces = postprocessing.get("Dielectric", ())
    if not isinstance(interfaces, (list, tuple)):
        return ()
    return tuple(entry for entry in interfaces if isinstance(entry, dict))


def _load_optional_postprocessing_index_map(
    source: str | Path | dict,
    *,
    index_map_path: str | Path | None,
) -> PostprocessingIndexMap | None:
    resolved_index_map_path = _find_optional_postprocessing_index_map_path(
        source,
        index_map_path=index_map_path,
    )
    if resolved_index_map_path is None:
        return None
    return load_postprocessing_index_map(
        source,
        index_map_path=resolved_index_map_path,
    )


def _load_optional_domain_material_index_map(
    source: str | Path | dict,
    *,
    index_map_path: str | Path | None,
) -> PostprocessingIndexMap | None:
    return _load_optional_postprocessing_index_map(
        source,
        index_map_path=index_map_path,
    )


def _load_optional_material_resolution_rows(
    source: str | Path | dict,
    *,
    material_resolution_path: str | Path | None,
) -> tuple[dict[str, Any], ...]:
    resolved_path = _find_optional_material_resolution_path(
        source,
        material_resolution_path=material_resolution_path,
    )
    if resolved_path is None or not resolved_path.exists():
        return ()
    data = json.loads(resolved_path.read_text())
    rows = data.get("materials", ()) if isinstance(data, dict) else ()
    if not isinstance(rows, (list, tuple)):
        return ()
    return tuple(dict(row) for row in rows if isinstance(row, dict))


def _load_optional_interface_material_resolution_rows(
    source: str | Path | dict,
    *,
    material_resolution_path: str | Path | None,
) -> tuple[dict[str, Any], ...]:
    resolved_path = _find_optional_material_resolution_path(
        source,
        material_resolution_path=material_resolution_path,
    )
    if resolved_path is None or not resolved_path.exists():
        return ()
    data = json.loads(resolved_path.read_text())
    rows = data.get("interfaces", ()) if isinstance(data, dict) else ()
    if not isinstance(rows, (list, tuple)):
        return ()
    return tuple(dict(row) for row in rows if isinstance(row, dict))


def _domain_material_row(
    *,
    material_row_index: int,
    material_attribute: int | None,
    material_attributes: tuple[int, ...],
    material: dict[str, Any],
    index_entry: Any | None,
    material_resolution: dict[str, Any] | None,
) -> dict[str, Any]:
    physical_name = None
    entry_name = None
    role = None
    section = None
    domain_index = None
    attributes = (material_attribute,) if material_attribute is not None else ()
    metadata: dict[str, Any] = {}
    source_name = (
        f"Attribute {material_attribute}"
        if material_attribute is not None
        else f"Material row {material_row_index}"
    )

    if index_entry is not None:
        domain_index = index_entry.index
        section = index_entry.section
        entry_name = index_entry.entry_name
        role = index_entry.role
        attributes = index_entry.attributes
        physical_name = (
            index_entry.physical_names[0] if index_entry.physical_names else None
        )
        source_name = index_entry.primary_physical_name
        metadata = dict(index_entry.metadata)

    return {
        "material_row_index": material_row_index,
        "material_attribute": material_attribute,
        "material_attributes": material_attributes,
        "domain_index": domain_index,
        "section": section,
        "source_name": source_name,
        "physical_name": physical_name,
        "entry_name": entry_name,
        "role": role,
        "attributes": attributes,
        "metadata": metadata,
        "material_name": _config_material_value(material, "Name", "Material"),
        "permittivity": _optional_numeric(
            _config_material_value(
                material,
                "Permittivity",
                "RelativePermittivity",
                "epsilon_r",
                "eps_r",
            )
        ),
        "loss_tangent": _optional_numeric(
            _config_material_value(
                material,
                "LossTan",
                "LossTangent",
                "loss_tangent",
                "tan_delta",
            )
        ),
        "conductivity": _optional_numeric(
            _config_material_value(
                material,
                "Conductivity",
                "conductivity",
            )
        ),
        "permeability": _optional_numeric(
            _config_material_value(
                material,
                "Permeability",
                "RelativePermeability",
                "mu_r",
            )
        ),
        "material_axes": _config_material_value(
            material,
            "MaterialAxes",
            "Axes",
        ),
        "volume_name": _material_resolution_value(material_resolution, "volume_name"),
        "stack_material_name": _material_resolution_value(
            material_resolution,
            "stack_material_name",
        ),
        "matched_material_name": _material_resolution_value(
            material_resolution,
            "matched_material_name",
        ),
        "material_model_type": _material_resolution_value(
            material_resolution,
            "model_type",
        ),
        "material_model_source": _material_resolution_value(
            material_resolution,
            "model_source",
        ),
        "material_within_validity": _material_resolution_value(
            material_resolution,
            "within_validity",
        ),
        "material_validity_note": _material_resolution_value(
            material_resolution,
            "validity_note",
        ),
        "material_frequency_hz": _optional_numeric(
            _material_resolution_value(material_resolution, "evaluation_frequency_hz")
        ),
        "material_frequency_ghz": _optional_numeric(
            _material_resolution_value(material_resolution, "evaluation_frequency_ghz")
        ),
        "raw_material_resolution": (
            {} if material_resolution is None else dict(material_resolution)
        ),
        "raw_material": dict(material),
    }


def _matching_material_resolution_row(
    rows: tuple[dict[str, Any], ...],
    *,
    material_row_index: int,
    material_attribute: int | None,
) -> dict[str, Any] | None:
    if material_attribute is not None:
        for row in rows:
            if _optional_int(row.get("material_attribute")) == material_attribute:
                return row

    for row in rows:
        if _optional_int(row.get("material_row_index")) == material_row_index:
            return row
    return None


def _matching_interface_material_resolution_row(
    rows: tuple[dict[str, Any], ...],
    *,
    interface_row_index: int,
    surface_index: int | None,
) -> dict[str, Any] | None:
    if surface_index is not None:
        for row in rows:
            if _optional_int(row.get("surface_index")) == surface_index:
                return row

    for row in rows:
        if _optional_int(row.get("interface_row_index")) == interface_row_index:
            return row
    return None


def _material_resolution_value(
    material_resolution: dict[str, Any] | None,
    key: str,
) -> Any:
    if material_resolution is None:
        return None
    return material_resolution.get(key)


def _material_attributes(material: dict[str, Any]) -> tuple[int, ...]:
    value = _config_material_value(material, "Attributes", "attributes")
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (int(value),)
    if isinstance(value, (int, float)):
        return (int(value),)
    return tuple(int(attribute) for attribute in value)


def _config_material_value(material: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in material:
            return material[key]
    lower_keys = {str(key).lower(): value for key, value in material.items()}
    for key in keys:
        value = lower_keys.get(key.lower())
        if value is not None:
            return value
    return None


def _optional_numeric(value: Any) -> Any:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _dielectric_interface_matches(
    index_map: PostprocessingIndexMap | None,
    *,
    surface_index: int | None,
    attributes: tuple[int, ...],
) -> tuple[Any, ...]:
    if index_map is None:
        return ()

    section = "Boundaries.Postprocessing.Dielectric"
    if surface_index is not None:
        entry = index_map.entry_for_index(section, surface_index)
        if entry is not None:
            return (entry,)

    matched_entries = []
    seen: set[tuple[str, int]] = set()
    for attribute in attributes:
        for entry in index_map.entries_for_attribute(attribute, section=section):
            key = (entry.section, entry.index)
            if key in seen:
                continue
            seen.add(key)
            matched_entries.append(entry)
    return tuple(matched_entries)


def _dielectric_interface_row(
    *,
    interface_row_index: int,
    surface_index: int | None,
    surface_attribute: int | None,
    surface_attributes: tuple[int, ...],
    interface: dict[str, Any],
    index_entry: Any | None,
    material_resolution: dict[str, Any] | None,
) -> dict[str, Any]:
    physical_name = None
    entry_name = None
    role = None
    section = (
        "Boundaries.Postprocessing.Dielectric" if surface_index is not None else None
    )
    attributes = surface_attributes
    metadata: dict[str, Any] = {}
    preset_name = None
    preset_source = None
    source_name = (
        f"Surface {surface_index}"
        if surface_index is not None
        else f"Interface row {interface_row_index}"
    )

    if index_entry is not None:
        surface_index = index_entry.index
        section = index_entry.section
        entry_name = index_entry.entry_name
        role = index_entry.role
        attributes = index_entry.attributes
        physical_name = (
            index_entry.physical_names[0] if index_entry.physical_names else None
        )
        source_name = index_entry.primary_physical_name
        metadata = dict(index_entry.metadata)
        preset_name = _optional_str(index_entry.extra.get("preset_name"))
        preset_source = _optional_str(index_entry.extra.get("preset_source"))

    return {
        "interface_row_index": interface_row_index,
        "surface_index": surface_index,
        "surface_attribute": surface_attribute,
        "surface_attributes": surface_attributes,
        "section": section,
        "source_name": source_name,
        "physical_name": physical_name,
        "entry_name": entry_name,
        "role": role,
        "attributes": attributes,
        "metadata": metadata,
        "interface_type": _config_material_value(interface, "Type", "interface_type"),
        "preset_name": preset_name,
        "preset_source": preset_source,
        "thickness": _optional_numeric(
            _config_material_value(interface, "Thickness", "thickness")
        ),
        "permittivity": _optional_numeric(
            _config_material_value(interface, "Permittivity", "permittivity")
        ),
        "loss_tangent": _optional_numeric(
            _config_material_value(
                interface,
                "LossTan",
                "LossTangent",
                "loss_tangent",
                "tan_delta",
            )
        ),
        "interface_material_name": _material_resolution_value(
            material_resolution,
            "interface_material_name",
        ),
        "matched_material_name": _material_resolution_value(
            material_resolution,
            "matched_material_name",
        ),
        "material_model_type": _material_resolution_value(
            material_resolution,
            "model_type",
        ),
        "material_model_source": _material_resolution_value(
            material_resolution,
            "model_source",
        ),
        "material_within_validity": _material_resolution_value(
            material_resolution,
            "within_validity",
        ),
        "material_validity_note": _material_resolution_value(
            material_resolution,
            "validity_note",
        ),
        "material_frequency_hz": _optional_numeric(
            _material_resolution_value(material_resolution, "evaluation_frequency_hz")
        ),
        "material_frequency_ghz": _optional_numeric(
            _material_resolution_value(material_resolution, "evaluation_frequency_ghz")
        ),
        "raw_material_resolution": (
            {} if material_resolution is None else dict(material_resolution)
        ),
        "raw_interface": dict(interface),
    }


def _matching_surface_attribute(
    index_entry: Any, attributes: tuple[int, ...]
) -> int | None:
    for attribute in attributes:
        if attribute in index_entry.attributes:
            return attribute
    return attributes[0] if attributes else None


def _load_eigenmode_history_for_report(
    source: str | Path | dict,
    eigenmodes: Eigenmodes,
    *,
    include_final: bool,
    source_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    import pandas as pd

    history_source = _eigenmode_history_source(source, eigenmodes.source_path)
    iteration_paths = _find_eigenmode_iteration_csvs(history_source)
    source_rows.append(
        _report_source_row(
            "iteration*/eig.csv",
            None,
            required=False,
            present=bool(iteration_paths),
            loaded=bool(iteration_paths),
            message=(
                f"loaded {len(iteration_paths)} AMR iteration files"
                if iteration_paths
                else "no AMR iteration eig.csv files found"
            ),
        )
    )

    if not include_final and not iteration_paths:
        return _empty_eigenmode_history()

    try:
        return load_eigenmode_history(history_source, include_final=include_final)
    except (FileNotFoundError, ValueError):
        if not include_final:
            return _empty_eigenmode_history()
        final_history = _eigenmodes_to_history_frame(
            eigenmodes,
            iteration_index=1,
            label="Final",
            is_final=True,
            source_kind="final",
            source_iteration=None,
        )
        return cast(
            "pd.DataFrame",
            _add_eigenmode_convergence_columns(pd.DataFrame(final_history)),
        )


def _load_terminal_matrix_for_report(
    source: str | Path | dict,
    matrix_kind: str,
    *,
    index_map_path: Path | None,
    terminal_names: tuple[str, ...] | list[str] | None,
    source_rows: list[dict[str, Any]],
    required: bool,
) -> TerminalMatrix | None:
    kind = _normalize_terminal_matrix_kind(matrix_kind)
    csv_name = str(_TERMINAL_MATRIX_SPECS[kind]["file_name"])
    csv_path = _find_optional_terminal_matrix_csv(source, kind)
    if csv_path is None or not csv_path.exists():
        source_rows.append(
            _report_source_row(
                csv_name,
                csv_path,
                required=required,
                present=False,
                loaded=False,
                message="not found",
            )
        )
        if required:
            msg = f"Required electrostatic {csv_name} not found"
            raise FileNotFoundError(msg)
        return None

    resolved_terminal_names = terminal_names
    if resolved_terminal_names is None and index_map_path is None:
        terminal_count = len(_read_terminal_matrix_csv(csv_path))
        resolved_terminal_names = tuple(
            f"T{index}" for index in range(1, terminal_count + 1)
        )

    matrix = load_terminal_matrix(
        {csv_name: csv_path, "palace_index_map.json": index_map_path}
        if index_map_path is not None
        else csv_path,
        kind,
        index_map_path=index_map_path,
        terminal_names=resolved_terminal_names,
    )
    source_rows.append(
        _report_source_row(
            csv_name,
            csv_path,
            required=required,
            present=True,
            loaded=True,
            message="loaded",
        )
    )
    return matrix


def _load_terminal_matrix_history_for_report(
    source: str | Path | dict,
    matrix: TerminalMatrix | None,
    *,
    include_history: bool,
    index_map_path: Path | None,
    terminal_names: tuple[str, ...] | list[str] | None,
    source_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    import pandas as pd

    if matrix is None:
        return _empty_terminal_matrix_history()

    csv_name = str(_TERMINAL_MATRIX_SPECS[matrix.matrix_kind]["file_name"])
    history_source = _terminal_matrix_history_source(source, matrix.source_path)
    iteration_paths = _find_terminal_matrix_iteration_csvs(
        history_source,
        matrix.matrix_kind,
    )
    source_rows.append(
        _report_source_row(
            f"iteration*/{csv_name}",
            None,
            required=False,
            present=bool(iteration_paths),
            loaded=include_history and bool(iteration_paths),
            message=(
                f"loaded {len(iteration_paths)} AMR iteration files"
                if include_history and iteration_paths
                else "history disabled"
                if not include_history
                else f"no AMR iteration {csv_name} files found"
            ),
        )
    )
    if not include_history:
        return _empty_terminal_matrix_history()

    try:
        return load_terminal_matrix_history(
            history_source,
            matrix.matrix_kind,
            index_map_path=index_map_path,
            terminal_names=terminal_names or matrix.terminal_names,
            include_final=True,
        )
    except (FileNotFoundError, ValueError):
        final_history = _terminal_matrix_to_history_frame(
            matrix,
            pass_index=1,
            label="Final",
            is_final=True,
        )
        return cast(
            "pd.DataFrame",
            _add_terminal_matrix_convergence_columns(pd.DataFrame(final_history)),
        )


def _terminal_matrix_history_source(source: str | Path | dict, csv_path: Path) -> Path:
    if isinstance(source, dict):
        return csv_path.parent

    path = Path(source)
    return path.parent if path.is_file() else path


def _find_terminal_matrix_iteration_csvs(
    source: str | Path,
    matrix_kind: str,
) -> tuple[Path, ...]:
    path = Path(source)
    if path.is_file() or not path.exists():
        return ()
    try:
        output_dir = _resolve_palace_output_dir(path)
    except FileNotFoundError:
        return ()
    csv_name = str(_TERMINAL_MATRIX_SPECS[matrix_kind]["file_name"])
    return tuple(
        iteration_dir / csv_name
        for iteration_dir, _ in _iteration_dirs(output_dir)
        if (iteration_dir / csv_name).exists()
    )


def _eigenmode_history_source(source: str | Path | dict, eig_csv_path: Path) -> Path:
    if isinstance(source, dict):
        return eig_csv_path.parent

    path = Path(source)
    return path.parent if path.is_file() else path


def _find_eigenmode_iteration_csvs(source: str | Path) -> tuple[Path, ...]:
    path = Path(source)
    if path.is_file() or not path.exists():
        return ()
    try:
        output_dir = _resolve_palace_output_dir(path)
    except FileNotFoundError:
        return ()
    return tuple(
        iteration_dir / "eig.csv"
        for iteration_dir, _ in _iteration_dirs(output_dir)
        if (iteration_dir / "eig.csv").exists()
    )


def _find_optional_postprocessing_index_map_path(
    source: str | Path | dict,
    *,
    index_map_path: str | Path | None,
) -> Path | None:
    if index_map_path is not None:
        return Path(index_map_path)
    return _find_postprocessing_index_map(source)


def _find_optional_config_path(
    source: str | Path | dict,
    *,
    config_path: str | Path | None,
) -> Path | None:
    if config_path is not None:
        return Path(config_path)
    return _find_config_json(source)


def _find_optional_material_resolution_path(
    source: str | Path | dict,
    *,
    material_resolution_path: str | Path | None,
) -> Path | None:
    if material_resolution_path is not None:
        return Path(material_resolution_path)
    return _find_material_resolution_json(source)


def _postprocessing_index_map_to_dataframe(
    index_map: PostprocessingIndexMap,
) -> pd.DataFrame:
    import pandas as pd

    rows = [
        {"schema_version": index_map.schema_version, **row}
        for row in index_map.to_rows()
    ]
    if not rows:
        return _empty_index_map_dataframe()
    return pd.DataFrame.from_records(rows)


def _load_optional_eigenmode_report_table(
    source: str | Path | dict,
    csv_name: str,
    *,
    loader: Any,
    empty_factory: Any,
    index_map_path: Path | None,
    index_map_present: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    csv_path = _find_optional_report_csv(source, csv_name)
    if csv_path is None or not csv_path.exists():
        return empty_factory(), _report_source_row(
            csv_name,
            csv_path,
            required=False,
            present=False,
            loaded=False,
            message="not found",
        )

    if not index_map_present:
        return empty_factory(), _report_source_row(
            csv_name,
            csv_path,
            required=False,
            present=True,
            loaded=False,
            message="missing palace_index_map.json",
        )

    report_source = {
        csv_name: csv_path,
        "palace_index_map.json": index_map_path,
    }
    loaded = loader(report_source, index_map_path=index_map_path)
    return loaded, _report_source_row(
        csv_name,
        csv_path,
        required=False,
        present=True,
        loaded=True,
        message="loaded",
    )


def _find_optional_report_csv(source: str | Path | dict, csv_name: str) -> Path | None:
    try:
        return _resolve_report_csv(source, csv_name)
    except (FileNotFoundError, ValueError):
        pass

    if isinstance(source, dict):
        for value in source.values():
            path = Path(value)
            root = path.parent if path.suffix else path
            found = _find_file(root, csv_name) if root.exists() else None
            if found is not None:
                return found
        return None

    path = Path(source)
    root = path.parent if path.is_file() else path
    if not root.exists():
        return None
    return _find_file(root, csv_name)


def _find_optional_terminal_matrix_csv(
    source: str | Path | dict,
    matrix_kind: str,
) -> Path | None:
    csv_name = str(_TERMINAL_MATRIX_SPECS[matrix_kind]["file_name"])
    if isinstance(source, dict):
        explicit = source.get(csv_name)
        if explicit is not None:
            return Path(explicit)
        for value in source.values():
            path = Path(value)
            root = path.parent if path.suffix else path
            found = _find_file(root, csv_name) if root.exists() else None
            if found is not None:
                return found
        return None

    path = Path(source)
    if path.is_file():
        if path.name == csv_name:
            return path
        root = path.parent
    else:
        root = path
    if not root.exists():
        return None
    return _find_file(root, csv_name)


def _empty_eigenmode_history() -> pd.DataFrame:
    import pandas as pd

    return pd.DataFrame(
        columns=[
            "iteration_index",
            "label",
            "is_final",
            "source_kind",
            "source_iteration",
            "source_path",
            "mode_index",
            "frequency_ghz",
            "imaginary_frequency_ghz",
            "q_factor",
            "backward_error",
            "absolute_error",
        ]
    )


def _empty_eigenmode_pass_summary() -> pd.DataFrame:
    import pandas as pd

    return pd.DataFrame(
        columns=[
            "iteration_index",
            "label",
            "is_final",
            "n_modes",
            "max_abs_delta_to_previous_mhz",
            "max_abs_relative_delta_to_previous_percent",
            "max_abs_delta_to_final_mhz",
            "max_abs_relative_delta_to_final_percent",
            "max_abs_imaginary_relative_delta_to_previous_percent",
            "hfss_max_delta_freq_percent",
        ]
    )


def _empty_terminal_matrix_history() -> pd.DataFrame:
    import pandas as pd

    return pd.DataFrame(
        columns=[
            "pass_index",
            "label",
            "is_final",
            "matrix_kind",
            "matrix_csv_path",
            "row_index",
            "column_index",
            "row_terminal",
            "column_terminal",
            "element",
            "is_diagonal",
            "value_si",
            "source_unit",
            "display_value",
            "display_unit",
            "display_scale",
            "delta_to_previous_si",
            "abs_delta_to_previous_si",
            "relative_delta_to_previous_percent",
            "abs_relative_delta_to_previous_percent",
            "delta_to_final_si",
            "abs_delta_to_final_si",
            "relative_delta_to_final_percent",
            "abs_relative_delta_to_final_percent",
            "display_delta_to_previous",
            "abs_display_delta_to_previous",
            "display_delta_to_final",
            "abs_display_delta_to_final",
        ]
    )


def _empty_terminal_matrix_pass_summary() -> pd.DataFrame:
    import pandas as pd

    return pd.DataFrame(
        columns=[
            "matrix_kind",
            "pass_index",
            "label",
            "is_final",
            "display_unit",
            "n_elements",
            "n_diagonal_elements",
            "max_abs_value",
            "max_abs_display_value",
            "max_abs_delta_to_previous",
            "max_abs_display_delta_to_previous",
            "max_abs_relative_delta_to_previous_percent",
            "max_abs_delta_to_final",
            "max_abs_display_delta_to_final",
            "max_abs_relative_delta_to_final_percent",
            "n_off_diagonal_elements",
        ]
    )


def _empty_index_map_dataframe() -> pd.DataFrame:
    import pandas as pd

    return pd.DataFrame(
        columns=[
            "schema_version",
            "section",
            "index",
            "entry_name",
            "role",
            "attributes",
            "physical_names",
            "dimension",
            "source",
            "interface_of",
            "exterior_of",
            "metadata",
        ]
    )


def _empty_domain_material_summary() -> pd.DataFrame:
    import pandas as pd

    return pd.DataFrame(columns=_DOMAIN_MATERIAL_COLUMNS)


def _empty_dielectric_interface_summary() -> pd.DataFrame:
    import pandas as pd

    return pd.DataFrame(columns=_DIELECTRIC_INTERFACE_COLUMNS)


def _empty_domain_energy_summary() -> pd.DataFrame:
    import pandas as pd

    return pd.DataFrame(
        columns=[
            "row_index",
            "sample_column",
            "sample_value",
            "mode_index",
            "domain_index",
            "section",
            "source_name",
            "physical_name",
            "entry_name",
            "role",
            "attributes",
            "E_elec_j",
            "E_mag_j",
            "p_elec",
            "p_mag",
        ]
    )


def _empty_surface_q_summary() -> pd.DataFrame:
    import pandas as pd

    return pd.DataFrame(
        columns=[
            "row_index",
            "sample_column",
            "sample_value",
            "mode_index",
            "surface_index",
            "section",
            "source_name",
            "physical_name",
            "entry_name",
            "role",
            "attributes",
            "interface_type",
            "p_surf",
            "q_surf",
            "inverse_q",
        ]
    )


def _empty_domain_loss_summary() -> pd.DataFrame:
    import pandas as pd

    return pd.DataFrame(columns=_DOMAIN_LOSS_COLUMNS)


def _empty_surface_loss_summary() -> pd.DataFrame:
    import pandas as pd

    return pd.DataFrame(columns=_SURFACE_LOSS_COLUMNS)


def _empty_loss_budget_summary() -> pd.DataFrame:
    import pandas as pd

    return pd.DataFrame(columns=_LOSS_BUDGET_COLUMNS)


def _empty_port_epr_summary() -> pd.DataFrame:
    import pandas as pd

    return pd.DataFrame(
        columns=[
            "row_index",
            "sample_column",
            "sample_value",
            "mode_index",
            "port_index",
            "section",
            "source_name",
            "physical_name",
            "entry_name",
            "role",
            "attributes",
            "postprocessing_type",
            "p_port",
            "abs_p_port",
            "abs_p_port_fraction",
        ]
    )


def _load_indexed_quantity_summary(
    source: str | Path | dict,
    csv_name: str,
    *,
    index_name: str,
    quantity_columns: dict[str, str],
    index_map_path: str | Path | None,
) -> pd.DataFrame:
    import pandas as pd

    indexed = load_indexed_csv(
        source,
        csv_name,
        index_map_path=index_map_path,
        rename_columns=False,
    )
    frame = indexed.dataframe
    indexed_column_names = {column.original_name for column in indexed.columns}
    sample_columns = [
        column for column in frame.columns if column not in indexed_column_names
    ]

    rows: dict[tuple[int, int], dict[str, Any]] = {}
    for row_offset, (_, source_row) in enumerate(frame.iterrows(), start=1):
        sample_metadata = _sample_metadata(source_row, sample_columns, row_offset)
        for column in indexed.columns:
            output_column = quantity_columns.get(column.quantity)
            if output_column is None:
                continue
            key = (row_offset, column.index)
            row = rows.setdefault(
                key,
                {
                    **sample_metadata,
                    **_indexed_column_provenance(column, index_name=index_name),
                },
            )
            row[output_column] = source_row[column.original_name]

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame.from_records(list(rows.values()))
    sort_columns = [
        column
        for column in (
            "mode_index",
            "source_index",
            "frequency_ghz",
            "row_index",
            index_name,
        )
        if column in result.columns
    ]
    if sort_columns:
        result = result.sort_values(sort_columns).reset_index(drop=True)
    return result


def _sample_metadata(
    source_row: Any, sample_columns: list[str], row_offset: int
) -> dict[str, Any]:
    row: dict[str, Any] = {"row_index": row_offset}
    if not sample_columns:
        return row

    sample_column = sample_columns[0]
    sample_value = source_row[sample_column]
    row["sample_column"] = sample_column
    row["sample_value"] = sample_value

    alias = _sample_column_alias(sample_column)
    if alias is not None:
        row[alias] = _coerce_sample_value(
            sample_value, integer=alias.endswith("_index")
        )
    return row


def _sample_column_alias(column: str) -> str | None:
    normalized = column.strip().lower()
    if normalized in {"m", "mode"}:
        return "mode_index"
    if normalized == "i":
        return "source_index"
    if normalized.startswith("f") and "ghz" in normalized:
        return "frequency_ghz"
    return None


def _coerce_sample_value(value: Any, *, integer: bool) -> Any:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    if integer and np.isfinite(numeric):
        return round(numeric)
    return numeric


def _indexed_column_provenance(
    column: IndexedCsvColumn,
    *,
    index_name: str,
) -> dict[str, Any]:
    source_name = column.physical_name or column.entry_name or f"Index {column.index}"
    row: dict[str, Any] = {
        index_name: column.index,
        "section": column.section,
        "source_name": source_name,
        "physical_name": column.physical_name,
        "entry_name": column.entry_name,
        "role": column.role,
        "attributes": column.attributes,
    }
    if column.metadata:
        row["metadata"] = dict(column.metadata)
    interface_type = _interface_type_for_column(column)
    if interface_type is not None:
        row["interface_type"] = interface_type
    postprocessing_type = column.extra.get("Type")
    if postprocessing_type is not None and interface_type is None:
        row["postprocessing_type"] = postprocessing_type
    return row


def _interface_type_for_column(column: IndexedCsvColumn) -> str | None:
    entry_type = column.extra.get("Type")
    if isinstance(entry_type, str) and entry_type in {"MA", "MS", "SA", "Default"}:
        return entry_type

    for name in (column.physical_name, column.entry_name):
        if not name:
            continue
        prefix = name.split(":", 1)[0]
        if prefix in {"MA", "MS", "SA"}:
            return prefix
    return None


def _inverse_q_values(values: Any) -> pd.Series:
    import pandas as pd

    numeric = cast("pd.Series", pd.to_numeric(values, errors="coerce")).fillna(
        float("nan")
    )
    return numeric.map(lambda value: _inverse_q_from_q(_float_or_nan(value)))


def _inverse_q_from_q(q_value: float) -> float:
    if not np.isfinite(q_value):
        return 0.0
    if q_value == 0.0:
        return float("inf")
    return 1.0 / q_value


def _q_from_inverse_q(inverse_q: float) -> float:
    if inverse_q == 0.0:
        return float("inf")
    if not np.isfinite(inverse_q):
        return 0.0
    return 1.0 / inverse_q


def _fraction(value: float, total: float) -> float:
    if total == 0.0 or not np.isfinite(total):
        return 0.0
    return value / total


def _fraction_values(values: Any, totals: Any) -> pd.Series:
    import pandas as pd

    numeric_values = cast("pd.Series", pd.to_numeric(values, errors="coerce")).fillna(
        0.0
    )
    if np.isscalar(totals):
        total = _float_or_nan(totals)
        if total == 0.0 or not np.isfinite(total):
            return numeric_values * 0.0
        return numeric_values / total

    numeric_totals = cast("pd.Series", pd.to_numeric(totals, errors="coerce")).fillna(
        float("nan")
    )
    result = numeric_values / numeric_totals
    return result.where((numeric_totals != 0.0) & np.isfinite(numeric_totals), 0.0)


def _matching_domain_material_row(
    domain_row: Any,
    domain_materials: pd.DataFrame,
) -> dict[str, Any] | None:
    if domain_materials.empty:
        return None

    domain_index = _optional_int(domain_row.get("domain_index"))
    if domain_index is not None and "domain_index" in domain_materials.columns:
        for _, material_row in domain_materials.iterrows():
            if _optional_int(material_row.get("domain_index")) == domain_index:
                return dict(material_row)

    attributes = _tuple_of_ints(domain_row.get("attributes"))
    if not attributes:
        return None

    if "material_attribute" in domain_materials.columns:
        for _, material_row in domain_materials.iterrows():
            attribute = _optional_int(material_row.get("material_attribute"))
            if attribute in attributes:
                return dict(material_row)

    if "material_attributes" in domain_materials.columns:
        wanted = set(attributes)
        for _, material_row in domain_materials.iterrows():
            material_attributes = set(
                _tuple_of_ints(material_row.get("material_attributes"))
            )
            if wanted & material_attributes:
                return dict(material_row)

    return None


def _matching_dielectric_interface_row(
    surface_row: Any,
    dielectric_interfaces: pd.DataFrame,
) -> dict[str, Any] | None:
    if dielectric_interfaces.empty:
        return None

    surface_index = _optional_int(surface_row.get("surface_index"))
    if surface_index is not None and "surface_index" in dielectric_interfaces.columns:
        for _, interface_row in dielectric_interfaces.iterrows():
            if _optional_int(interface_row.get("surface_index")) == surface_index:
                return dict(interface_row)

    attributes = _tuple_of_ints(surface_row.get("attributes"))
    if not attributes:
        return None

    if "surface_attribute" in dielectric_interfaces.columns:
        for _, interface_row in dielectric_interfaces.iterrows():
            attribute = _optional_int(interface_row.get("surface_attribute"))
            if attribute in attributes:
                return dict(interface_row)

    if "surface_attributes" in dielectric_interfaces.columns:
        wanted = set(attributes)
        for _, interface_row in dielectric_interfaces.iterrows():
            interface_attributes = set(
                _tuple_of_ints(interface_row.get("surface_attributes"))
            )
            if wanted & interface_attributes:
                return dict(interface_row)

    return None


def _ordered_dataframe(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    ordered = [column for column in columns if column in frame.columns]
    extra = [column for column in frame.columns if column not in ordered]
    return cast("pd.DataFrame", frame.loc[:, [*ordered, *extra]])


def _frequency_for_mode(
    mode_index: int | None,
    *,
    modes: Eigenmodes | pd.DataFrame | None,
    frequency_ghz: float | None,
) -> float | None:
    if frequency_ghz is not None:
        return float(frequency_ghz)
    if mode_index is None or modes is None:
        return None

    frame = modes.dataframe if isinstance(modes, Eigenmodes) else modes
    if frame.empty or "mode_index" not in frame.columns:
        return None
    frequency_column = (
        "freq_real_ghz"
        if "freq_real_ghz" in frame.columns
        else "frequency_ghz"
        if "frequency_ghz" in frame.columns
        else None
    )
    if frequency_column is None:
        return None

    matches = frame.loc[frame["mode_index"].map(_optional_int) == mode_index]
    if matches.empty:
        return None
    frequency = _float_or_nan(matches.iloc[0][frequency_column])
    return frequency if np.isfinite(frequency) else None


def _q_for_mode(
    mode_index: int | None,
    *,
    modes: Eigenmodes | pd.DataFrame | None,
) -> float:
    if mode_index is None or modes is None:
        return float("nan")
    frame = modes.dataframe if isinstance(modes, Eigenmodes) else modes
    if frame.empty or "mode_index" not in frame.columns:
        return float("nan")
    q_column = (
        "q"
        if "q" in frame.columns
        else "q_factor"
        if "q_factor" in frame.columns
        else None
    )
    if q_column is None:
        return float("nan")

    matches = frame.loc[frame["mode_index"].map(_optional_int) == mode_index]
    if matches.empty:
        return float("nan")
    return _float_or_nan(matches.iloc[0][q_column])


def _loss_mode_indices(
    domain_loss: pd.DataFrame,
    surface_loss: pd.DataFrame,
    modes: Eigenmodes | pd.DataFrame | None,
) -> tuple[int | None, ...]:
    mode_indices: set[int] = set()
    for frame in (domain_loss, surface_loss):
        if "mode_index" not in frame.columns:
            continue
        for value in frame["mode_index"]:
            mode_index = _optional_int(value)
            if mode_index is not None:
                mode_indices.add(mode_index)

    if modes is not None:
        frame = modes.dataframe if isinstance(modes, Eigenmodes) else modes
        if "mode_index" in frame.columns:
            for value in frame["mode_index"]:
                mode_index = _optional_int(value)
                if mode_index is not None:
                    mode_indices.add(mode_index)

    return tuple(sorted(mode_indices))


def _rows_for_mode(frame: pd.DataFrame, mode_index: int | None) -> pd.DataFrame:
    if frame.empty:
        return frame
    if mode_index is None or "mode_index" not in frame.columns:
        return frame
    return cast(
        "pd.DataFrame", frame.loc[frame["mode_index"].map(_optional_int) == mode_index]
    )


def _rate_columns_for_frequency(
    *,
    frequency_ghz: float | None,
    inverse_q: float,
) -> dict[str, float]:
    if frequency_ghz is None:
        return {}
    frequency = float(frequency_ghz)
    if frequency <= 0.0 or not np.isfinite(frequency):
        return {}
    gamma_hz = frequency * 1.0e9 * inverse_q
    gamma_rad_per_s = 2.0 * np.pi * gamma_hz
    return {
        "gamma_rad_per_s": gamma_rad_per_s,
        "gamma_per_us": gamma_rad_per_s / 1.0e6,
        "gamma_hz": gamma_hz,
        "gamma_mhz": gamma_hz / 1.0e6,
        "t1_us": float("inf") if gamma_rad_per_s <= 0.0 else 1.0e6 / gamma_rad_per_s,
    }


def _validate_positive_frequency_ghz(frequency_ghz: float) -> None:
    frequency = float(frequency_ghz)
    if frequency <= 0.0 or not np.isfinite(frequency):
        msg = f"frequency_ghz must be a positive finite value, got {frequency_ghz!r}"
        raise ValueError(msg)


def _numeric_or_default(value: Any, *, default: float) -> float:
    numeric = _float_or_nan(value)
    return numeric if np.isfinite(numeric) else default


def _tuple_of_ints(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        try:
            return (int(value),)
        except ValueError:
            return ()
    if isinstance(value, (int, float)):
        if np.isfinite(float(value)):
            return (int(value),)
        return ()
    try:
        return tuple(int(item) for item in value)
    except (TypeError, ValueError):
        return ()


def _relative_error(estimate: float, reference: float) -> float:
    if reference == 0.0 or not np.isfinite(reference):
        return float("nan")
    return (estimate - reference) / reference


def _sum_numeric_column(frame: pd.DataFrame, column: str) -> float:
    import pandas as pd

    if column not in frame.columns:
        return 0.0
    values = cast("pd.Series", pd.to_numeric(frame[column], errors="coerce")).fillna(
        0.0
    )
    return _float_or_nan(values.sum())


def _float_or_nan(value: Any) -> float:
    try:
        return float(cast("Any", value))
    except (TypeError, ValueError):
        return float("nan")


def _sample_group_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in ("mode_index", "source_index", "frequency_ghz", "sample_value")
        if column in frame.columns
    ]


def _parse_sparam_col(col: str) -> tuple[int, int, str] | None:
    """Parse a Palace S-parameter column header.

    Returns (i, j, "db"|"deg") or None.
    """
    # |S[i][j]| (dB)
    m = re.match(r"\|S\[(\d+)\]\[(\d+)\]\|\s*\((\w+\.?)\)", col)
    if m:
        return int(m.group(1)), int(m.group(2)), "db"

    # arg(S[i][j]) (deg.)
    m = re.match(r"arg\(S\[(\d+)\]\[(\d+)\]\)\s*\((\w+\.?)\)", col)
    if m:
        return int(m.group(1)), int(m.group(2)), "deg"

    return None


def _parse_indexed_csv_column(col: str) -> tuple[str, int, str] | None:
    """Parse ``quantity[index] suffix`` Palace report columns."""
    match = _INDEXED_CSV_COLUMN_RE.fullmatch(col.strip())
    if match is None:
        return None
    return (
        match.group("prefix"),
        int(match.group("index")),
        match.group("suffix"),
    )


def _parse_terminal_matrix_column_index(column: str) -> int:
    match = _TERMINAL_MATRIX_COLUMN_RE.search(column.strip())
    if match is None:
        msg = f"Could not parse Palace electrostatic matrix column: {column!r}"
        raise ValueError(msg)
    return int(match.group("index"))


def _section_for_csv(csv_name: str) -> str:
    try:
        return _CSV_INDEX_SECTIONS[csv_name]
    except KeyError:
        msg = (
            f"Cannot infer Palace index-map section for {csv_name!r}; "
            "pass section= explicitly."
        )
        raise ValueError(msg) from None


def _resolve_source(
    source: str | Path | dict,
    *,
    require_csv: bool = True,
) -> tuple[Path | None, Path]:
    """Turn *source* into ``(csv_path, base_dir)``."""
    if isinstance(source, dict):
        csv_val = source.get("port-S.csv")
        if csv_val is not None:
            csv_path = Path(csv_val)
            return csv_path, csv_path.parent
        for val in source.values():
            p = Path(val)
            if p.exists():
                return None, p.parent
        if require_csv:
            msg = "Results dict has no 'port-S.csv' entry"
            raise FileNotFoundError(msg)
        return None, Path()

    output_dir = Path(source)
    csv_path = _find_file(output_dir, "port-S.csv")
    if csv_path is None and require_csv:
        msg = f"port-S.csv not found in {output_dir} or its subdirectories"
        raise FileNotFoundError(msg)
    return csv_path, output_dir


def _load_terminal_matrix_from_csv(
    csv_path: Path,
    matrix_kind: str,
    *,
    label_source: str | Path | dict,
    index_map_path: str | Path | None,
    terminal_names: tuple[str, ...] | list[str] | None,
    display_scale: float | None,
    display_unit: str | None,
) -> TerminalMatrix:
    spec = _TERMINAL_MATRIX_SPECS[matrix_kind]
    raw = _read_terminal_matrix_csv(csv_path)
    labels = _resolve_terminal_matrix_labels(
        label_source,
        raw.shape[0],
        index_map_path=index_map_path,
        terminal_names=terminal_names,
    )
    matrix = cast("pd.DataFrame", raw.copy())
    matrix.index = labels
    matrix.columns = labels
    matrix.attrs.update(
        {
            "matrix_kind": matrix_kind,
            "source_unit": spec["source_unit"],
            "csv_path": str(csv_path),
        }
    )
    return TerminalMatrix(
        source_path=csv_path,
        matrix_kind=matrix_kind,
        dataframe=matrix,
        terminal_names=tuple(labels),
        source_unit=spec["source_unit"],
        display_scale=(
            float(spec["display_scale"])
            if display_scale is None
            else float(display_scale)
        ),
        display_unit=spec["display_unit"] if display_unit is None else display_unit,
    )


def _resolve_terminal_matrix_csv(source: str | Path | dict, matrix_kind: str) -> Path:
    csv_name = str(_TERMINAL_MATRIX_SPECS[matrix_kind]["file_name"])
    if isinstance(source, dict):
        csv_val = source.get(csv_name)
        if csv_val is None:
            msg = f"Results dict has no {csv_name!r} entry"
            raise FileNotFoundError(msg)
        return Path(csv_val)

    path = Path(source)
    if path.is_file():
        return path
    found = _find_file(path, csv_name)
    if found is None:
        msg = f"{csv_name} not found in {path} or its subdirectories"
        raise FileNotFoundError(msg)
    return found


def _resolve_eig_csv(source: str | Path | dict) -> Path:
    if isinstance(source, dict):
        explicit = source.get("eig.csv")
        if explicit is not None:
            return Path(explicit)
        matches = [
            Path(value)
            for name, value in source.items()
            if str(name).endswith(".csv") and _EIG_CSV_RE.fullmatch(str(name))
        ]
        if matches:
            return matches[0]
        msg = f"Results dict has no eigenmode CSV entry: {list(source)}"
        raise FileNotFoundError(msg)

    path = Path(source)
    if path.is_file():
        return path
    found = _find_eig_csv(path)
    if found is None:
        msg = f"eig.csv not found in {path} or its subdirectories"
        raise FileNotFoundError(msg)
    return found


def _find_eig_csv(base: Path) -> Path | None:
    candidates = [
        base / "eig.csv",
        base / "output" / "palace" / "eig.csv",
        base / "palace" / "eig.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if not base.exists():
        return None
    matches = [path for path in base.rglob("*.csv") if _EIG_CSV_RE.fullmatch(path.name)]
    return matches[0] if matches else None


def _find_eigenmode_column(
    columns: list[str] | tuple[str, ...],
    patterns: tuple[str, ...],
    *,
    default: str | None,
) -> str | None:
    normalized_patterns = tuple(pattern.lower() for pattern in patterns)
    for column in columns:
        lower = column.lower().strip()
        if any(pattern in lower for pattern in normalized_patterns):
            return column
    return default


def _resolve_palace_output_dir(base: Path) -> Path:
    candidates = [
        base,
        base / "output" / "palace",
        base / "palace",
    ]
    for candidate in candidates:
        if candidate.exists() and (
            (candidate / "eig.csv").exists()
            or any(
                path.is_dir() and _ITERATION_DIR_RE.fullmatch(path.name)
                for path in candidate.iterdir()
            )
        ):
            return candidate
    if not base.exists():
        raise FileNotFoundError(base)
    return base


def _eigenmodes_to_history_frame(
    eigenmodes: Eigenmodes,
    *,
    iteration_index: int,
    label: str,
    is_final: bool,
    source_kind: str,
    source_iteration: int | None,
) -> pd.DataFrame:
    frame = eigenmodes.to_report_dataframe()
    frame.insert(0, "source_path", str(eigenmodes.source_path))
    frame.insert(0, "source_iteration", source_iteration)
    frame.insert(0, "source_kind", source_kind)
    frame.insert(0, "is_final", is_final)
    frame.insert(0, "label", label)
    frame.insert(0, "iteration_index", iteration_index)
    return cast("pd.DataFrame", frame)


def _add_eigenmode_convergence_columns(history: pd.DataFrame) -> pd.DataFrame:
    frame = history.copy()
    if frame.empty:
        return frame

    frame = frame.sort_values(["mode_index", "iteration_index"]).reset_index(drop=True)
    grouped = frame.groupby("mode_index", sort=True)

    previous_frequency = grouped["frequency_ghz"].shift(1)
    final_frequency = grouped["frequency_ghz"].transform("last")
    previous_imaginary_frequency = grouped["imaginary_frequency_ghz"].shift(1)
    final_imaginary_frequency = grouped["imaginary_frequency_ghz"].transform("last")

    frame["delta_to_previous_ghz"] = frame["frequency_ghz"] - previous_frequency
    frame["delta_to_previous_mhz"] = frame["delta_to_previous_ghz"] * 1.0e3
    frame["abs_delta_to_previous_mhz"] = frame["delta_to_previous_mhz"].abs()
    frame["relative_delta_to_previous_percent"] = (
        frame["delta_to_previous_ghz"] / previous_frequency.abs()
    ) * 1.0e2
    frame["abs_relative_delta_to_previous_percent"] = frame[
        "relative_delta_to_previous_percent"
    ].abs()

    frame["delta_to_final_ghz"] = frame["frequency_ghz"] - final_frequency
    frame["delta_to_final_mhz"] = frame["delta_to_final_ghz"] * 1.0e3
    frame["abs_delta_to_final_mhz"] = frame["delta_to_final_mhz"].abs()
    frame["relative_delta_to_final_percent"] = (
        frame["delta_to_final_ghz"] / final_frequency.abs()
    ) * 1.0e2
    frame["abs_relative_delta_to_final_percent"] = frame[
        "relative_delta_to_final_percent"
    ].abs()

    frame["imaginary_delta_to_previous_ghz"] = (
        frame["imaginary_frequency_ghz"] - previous_imaginary_frequency
    )
    frame["imaginary_delta_to_previous_mhz"] = (
        frame["imaginary_delta_to_previous_ghz"] * 1.0e3
    )
    frame["abs_imaginary_delta_to_previous_mhz"] = frame[
        "imaginary_delta_to_previous_mhz"
    ].abs()
    frame["imaginary_relative_delta_to_previous_percent"] = (
        frame["imaginary_delta_to_previous_ghz"] / previous_imaginary_frequency.abs()
    ) * 1.0e2
    frame["abs_imaginary_relative_delta_to_previous_percent"] = frame[
        "imaginary_relative_delta_to_previous_percent"
    ].abs()

    frame["imaginary_delta_to_final_ghz"] = (
        frame["imaginary_frequency_ghz"] - final_imaginary_frequency
    )
    frame["imaginary_delta_to_final_mhz"] = (
        frame["imaginary_delta_to_final_ghz"] * 1.0e3
    )
    frame["abs_imaginary_delta_to_final_mhz"] = frame[
        "imaginary_delta_to_final_mhz"
    ].abs()
    frame["imaginary_relative_delta_to_final_percent"] = (
        frame["imaginary_delta_to_final_ghz"] / final_imaginary_frequency.abs()
    ) * 1.0e2
    frame["abs_imaginary_relative_delta_to_final_percent"] = frame[
        "imaginary_relative_delta_to_final_percent"
    ].abs()
    return frame.sort_values(["iteration_index", "mode_index"]).reset_index(drop=True)


def _next_eigenmode_iteration(pass_frames: list[pd.DataFrame]) -> int:
    max_index = 0
    for frame in pass_frames:
        max_index = max(max_index, int(cast("Any", frame["iteration_index"].max())))
    return max_index + 1


def _eigenmode_tables_match(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    columns = [
        "mode_index",
        "freq_real_ghz",
        "freq_imag_ghz",
        "q",
        "error_backward",
        "error_absolute",
    ]
    return (
        left.loc[:, columns]
        .reset_index(drop=True)
        .equals(right.loc[:, columns].reset_index(drop=True))
    )


def _find_terminal_matrix_final_csv(base: Path, matrix_kind: str) -> Path | None:
    csv_name = str(_TERMINAL_MATRIX_SPECS[matrix_kind]["file_name"])
    candidates = [
        base / csv_name,
        base / "output" / "palace" / csv_name,
        base / "palace" / csv_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _read_terminal_matrix_csv(csv_path: Path) -> pd.DataFrame:
    import pandas as pd

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    frame = pd.read_csv(csv_path, skipinitialspace=True)
    frame.columns = [str(column).strip() for column in frame.columns]
    if frame.empty or len(frame.columns) < 2:
        msg = f"Palace electrostatic matrix CSV is empty or incomplete: {csv_path}"
        raise ValueError(msg)

    row_column = frame.columns[0]
    row_indices = frame[row_column].astype(float).round().astype(int).tolist()
    matrix_columns = list(frame.columns[1:])
    column_indices = [
        _parse_terminal_matrix_column_index(column) for column in matrix_columns
    ]

    if len(set(row_indices)) != len(row_indices):
        msg = f"Duplicate Palace matrix row indices in {csv_path}: {row_indices}"
        raise ValueError(msg)
    if len(set(column_indices)) != len(column_indices):
        msg = f"Duplicate Palace matrix column indices in {csv_path}: {column_indices}"
        raise ValueError(msg)
    if len(row_indices) != len(column_indices):
        msg = (
            f"Palace electrostatic matrix must be square; found "
            f"{len(row_indices)} rows and {len(column_indices)} columns in {csv_path}"
        )
        raise ValueError(msg)

    expected_indices = set(range(1, len(row_indices) + 1))
    if set(row_indices) != expected_indices or set(column_indices) != expected_indices:
        msg = (
            "Palace electrostatic matrix indices must be contiguous and 1-based "
            f"in {csv_path}"
        )
        raise ValueError(msg)

    matrix = frame[matrix_columns].astype(float)
    matrix.index = row_indices
    matrix.columns = column_indices
    return cast(
        "pd.DataFrame", matrix.sort_index().reindex(sorted(column_indices), axis=1)
    )


def _terminal_matrix_to_history_frame(
    matrix: TerminalMatrix,
    *,
    pass_index: int,
    label: str,
    is_final: bool,
) -> pd.DataFrame:
    frame = matrix.to_long_dataframe()
    frame.insert(0, "is_final", is_final)
    frame.insert(0, "label", label)
    frame.insert(0, "pass_index", pass_index)
    return frame


def _add_terminal_matrix_convergence_columns(history: pd.DataFrame) -> pd.DataFrame:
    frame = history.copy()
    if frame.empty:
        return frame

    frame = frame.sort_values(
        ["matrix_kind", "row_index", "column_index", "pass_index"]
    ).reset_index(drop=True)
    grouped = frame.groupby(["matrix_kind", "row_index", "column_index"], sort=True)

    previous_value = grouped["value_si"].shift(1)
    final_value = grouped["value_si"].transform("last")
    display_scale = frame["display_scale"]

    frame["delta_to_previous_si"] = frame["value_si"] - previous_value
    frame["abs_delta_to_previous_si"] = frame["delta_to_previous_si"].abs()
    frame["relative_delta_to_previous_percent"] = (
        frame["delta_to_previous_si"] / previous_value.abs()
    ) * 1.0e2
    frame["abs_relative_delta_to_previous_percent"] = frame[
        "relative_delta_to_previous_percent"
    ].abs()

    frame["delta_to_final_si"] = frame["value_si"] - final_value
    frame["abs_delta_to_final_si"] = frame["delta_to_final_si"].abs()
    frame["relative_delta_to_final_percent"] = (
        frame["delta_to_final_si"] / final_value.abs()
    ) * 1.0e2
    frame["abs_relative_delta_to_final_percent"] = frame[
        "relative_delta_to_final_percent"
    ].abs()

    frame["display_delta_to_previous"] = frame["delta_to_previous_si"] * display_scale
    frame["abs_display_delta_to_previous"] = frame["display_delta_to_previous"].abs()
    frame["display_delta_to_final"] = frame["delta_to_final_si"] * display_scale
    frame["abs_display_delta_to_final"] = frame["display_delta_to_final"].abs()
    return frame.sort_values(["pass_index", "row_index", "column_index"]).reset_index(
        drop=True
    )


def _iteration_dirs(output_dir: Path) -> tuple[tuple[Path, int], ...]:
    if not output_dir.exists():
        raise FileNotFoundError(output_dir)
    dirs: list[tuple[Path, int]] = []
    for path in output_dir.iterdir():
        if not path.is_dir():
            continue
        match = _ITERATION_DIR_RE.fullmatch(path.name)
        if match is None:
            continue
        dirs.append((path, int(match.group(1))))
    return tuple(sorted(dirs, key=lambda item: item[1]))


def _next_pass_index(pass_frames: list[pd.DataFrame]) -> int:
    max_index = 0
    for frame in pass_frames:
        max_index = max(max_index, int(cast("Any", frame["pass_index"].max())))
    return max_index + 1


def _terminal_matrices_match(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    return left.equals(right)


def _resolve_terminal_matrix_labels(
    source: str | Path | dict,
    terminal_count: int,
    *,
    index_map_path: str | Path | None,
    terminal_names: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    if terminal_names is not None:
        labels = tuple(str(name) for name in terminal_names)
    else:
        labels = _terminal_names_from_index_map(
            source,
            terminal_count,
            index_map_path=index_map_path,
        )

    if len(labels) != terminal_count:
        msg = (
            f"Terminal label count ({len(labels)}) does not match Palace matrix "
            f"size ({terminal_count})."
        )
        raise ValueError(msg)
    if len(set(labels)) != len(labels):
        msg = f"Terminal labels must be unique: {labels!r}"
        raise ValueError(msg)
    return labels


def _terminal_names_from_index_map(
    source: str | Path | dict,
    terminal_count: int,
    *,
    index_map_path: str | Path | None,
) -> tuple[str, ...]:
    index_map = load_postprocessing_index_map(source, index_map_path=index_map_path)
    labels_by_index: dict[int, str] = {}
    for index in range(1, terminal_count + 1):
        entries = index_map.entry_for_index("Boundaries.Terminal", index)
        if entries is None:
            labels_by_index[index] = f"T{index}"
            continue
        terminal_name = entries.extra.get("terminal_name")
        labels_by_index[index] = (
            str(terminal_name)
            if terminal_name is not None
            else entries.primary_physical_name
        )
    return tuple(labels_by_index[index] for index in range(1, terminal_count + 1))


def _normalize_terminal_matrix_kind(matrix_kind: str) -> str:
    aliases = {
        "c": "C",
        "capacitance": "C",
        "mutual": "Cm",
        "cm": "Cm",
        "c_m": "Cm",
        "inverse": "Cinv",
        "cinv": "Cinv",
        "c_inv": "Cinv",
    }
    key = matrix_kind.strip()
    normalized = _TERMINAL_MATRIX_SPECS.get(key)
    if normalized is not None:
        return key
    alias = aliases.get(key.lower())
    if alias is not None:
        return alias
    allowed = ", ".join(_TERMINAL_MATRIX_SPECS)
    msg = (
        f"Unknown Palace terminal matrix kind {matrix_kind!r}; "
        f"expected one of {allowed}."
    )
    raise ValueError(msg)


def _resolve_report_csv(source: str | Path | dict, csv_name: str | None) -> Path:
    """Resolve a Palace report CSV from a direct path, directory, or result dict."""
    if isinstance(source, dict):
        if csv_name is None:
            candidates = [
                name for name in _CSV_INDEX_SECTIONS if source.get(name) is not None
            ]
            if len(candidates) != 1:
                msg = (
                    "Pass csv_name= when a results dict contains zero or multiple "
                    "indexed CSVs."
                )
                raise ValueError(msg)
            csv_name = candidates[0]
        csv_val = source.get(csv_name)
        if csv_val is None:
            msg = f"Results dict has no {csv_name!r} entry"
            raise FileNotFoundError(msg)
        return Path(csv_val)

    path = Path(source)
    if path.is_file():
        if csv_name is not None and path.name != csv_name:
            msg = f"CSV path {path} does not match csv_name={csv_name!r}"
            raise ValueError(msg)
        return path
    if csv_name is None:
        msg = "Pass csv_name= when source is a directory."
        raise ValueError(msg)
    found = _find_file(path, csv_name)
    if found is None:
        msg = f"{csv_name} not found in {path} or its subdirectories"
        raise FileNotFoundError(msg)
    return found


def _load_port_map(
    output_dir: Path,
    csv_path: Path | None,
    port_info_path: str | Path | None = None,
) -> dict[int, str]:
    """Build ``{port_number: port_name}`` from ``port_information.json``."""
    if port_info_path is not None:
        info_path = Path(port_info_path)
    else:
        info_path = _find_port_info(output_dir, csv_path)

    if info_path is None or not info_path.exists():
        logger.warning(
            "port_information.json not found — using numeric port names (p1, p2, …)"
        )
        return {}

    with open(info_path) as f:
        data = json.load(f)

    port_map: dict[int, str] = {}
    for entry in data.get("ports", []):
        num = entry.get("portnumber")
        name = entry.get("name")
        if num is not None and name is not None:
            port_map[num] = name
        elif num is not None:
            port_map[num] = f"p{num}"

    if port_map and all(
        v.startswith("p") and v[1:].isdigit() for v in port_map.values()
    ):
        logger.info(
            "port_information.json has no 'name' fields — "
            "columns will use numeric names (p1, p2, …). "
            "Re-mesh to get named columns."
        )

    return port_map


def _find_port_info(output_dir: Path, csv_path: Path | None) -> Path | None:
    """Search common locations for ``port_information.json``."""
    name = "port_information.json"
    candidates = [
        output_dir / name,
        output_dir / "output" / name,
    ]
    if csv_path is not None:
        candidates.insert(0, csv_path.parent / name)
        # Cloud layout: results/output/port-S.csv + results/input/port_info
        candidates.append(csv_path.parent.parent / "input" / name)
        candidates.append(csv_path.parent.parent / name)
        candidates.append(csv_path.parent.parent.parent / name)

    for p in candidates:
        if p.exists():
            return p
    return None


def _find_file(base: Path, name: str) -> Path | None:
    """Find *name* in *base* or common Palace subdirectories."""
    candidates = [
        base / name,
        base / "output" / "palace" / name,
        base / "palace" / name,
    ]
    for p in candidates:
        if p.exists():
            return p

    matches = list(base.rglob(name))
    return matches[0] if matches else None


def _find_postprocessing_index_map(source: str | Path | dict) -> Path | None:
    """Search common local/cloud locations for ``palace_index_map.json``."""
    name = "palace_index_map.json"
    if isinstance(source, dict):
        explicit = source.get(name)
        if explicit is not None:
            return Path(explicit)
        candidate_roots = [Path(value).parent for value in source.values()]
    else:
        path = Path(source)
        candidate_roots = [path if path.is_dir() else path.parent]

    for root in candidate_roots:
        candidates = [
            root / name,
            root.parent / name,
            root.parent.parent / name,
            root / "input" / name,
            root.parent / "input" / name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        if root.exists():
            found = _find_file(root, name)
            if found is not None:
                return found
    return None


def _find_config_json(source: str | Path | dict) -> Path | None:
    """Search common local/cloud locations for Palace ``config.json``."""
    name = "config.json"
    if isinstance(source, dict):
        explicit = source.get(name)
        if explicit is not None:
            return Path(explicit)
        candidate_roots = [Path(value).parent for value in source.values()]
    else:
        path = Path(source)
        candidate_roots = [path if path.is_dir() else path.parent]

    for root in candidate_roots:
        candidates = [
            root / name,
            root.parent / name,
            root.parent.parent / name,
            root / "input" / name,
            root.parent / "input" / name,
            root.parent.parent / "input" / name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        if root.exists():
            found = _find_file(root, name)
            if found is not None:
                return found
    return None


def _find_material_resolution_json(source: str | Path | dict) -> Path | None:
    """Search common local/cloud locations for material-resolution sidecars."""
    name = "palace_material_resolution.json"
    if isinstance(source, dict):
        explicit = source.get(name)
        if explicit is not None:
            return Path(explicit)
        candidate_roots = [Path(value).parent for value in source.values()]
    else:
        path = Path(source)
        candidate_roots = [path if path.is_dir() else path.parent]

    for root in candidate_roots:
        candidates = [
            root / name,
            root.parent / name,
            root.parent.parent / name,
            root / "input" / name,
            root.parent / "input" / name,
            root.parent.parent / "input" / name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        if root.exists():
            found = _find_file(root, name)
            if found is not None:
                return found
    return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_str_pair(value: Any) -> tuple[str, str] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    return str(value[0]), str(value[1])


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


# -----------------------------------------------------------------------
# Field loading
# -----------------------------------------------------------------------


def load_fields(
    source: str | Path | dict,
    *,
    excitation: int = 1,
    cycle: int | None = None,
    boundary: bool = False,
):
    """Load the ParaView volume or boundary dataset for a Palace simulation.

    Requires the simulation to have been run with ``save_step >= 1``
    so that field data was written to disk.

    Args:
        source: Results dict from ``sim.run_local()`` / ``sim.run()``,
            or a path to the simulation directory.
        excitation: Excitation index (1-based) to load.
        cycle: ParaView cycle number (``None`` -> last available).
        boundary: If ``True``, load boundary surface fields
            (``driven_boundary/``) instead of volume fields
            (``driven/``).  Boundary data includes ``J_s_real``,
            ``Q_s_real``, etc.

    Returns:
        ``pyvista.DataSet`` with point data such as
        ``E_real``, ``B_real``, ``S`` (volume) or
        ``J_s_real``, ``Q_s_real`` (boundary).

    Raises:
        FileNotFoundError: If paraview output is missing.

    Example::

        vol = load_fields(results)
        bnd = load_fields(results, boundary=True)
        plot_cross_section(vol, normal="x", origin=0)
    """
    import pyvista as pv

    _, base_dir = _resolve_source(source, require_csv=False)
    pvtu_path = _find_paraview_dir(base_dir, excitation, cycle, boundary=boundary)
    return pv.read(str(pvtu_path))


def _find_paraview_dir(
    base_dir: Path,
    excitation: int,
    cycle: int | None,
    *,
    boundary: bool = False,
) -> Path:
    """Locate the ``.pvtu`` file for the requested excitation and cycle.

    Palace uses two output layouts depending on the port type:
    - Multi-excitation (wave ports):  ``paraview/driven/excitation_N/CycleNNNNNN/``
    - Single-excitation (lumped ports): ``paraview/driven/CycleNNNNNN/`` (no subdir)
    Both are searched, with the explicit ``excitation_N`` folder taking priority.
    """
    subdir = "driven_boundary" if boundary else "driven"
    search_roots = [
        base_dir,
        base_dir / "output" / "palace",
    ]
    exc_dir: Path | None = None
    for root in search_roots:
        # Layout 1: explicit excitation subfolder (wave ports / multi-excitation)
        candidate = root / "paraview" / subdir / f"excitation_{excitation}"
        if candidate.is_dir():
            exc_dir = candidate
            break
        # Layout 2: flat — Cycle dirs sit directly under driven/ (lumped ports)
        flat = root / "paraview" / subdir
        if flat.is_dir() and any(flat.iterdir()):
            exc_dir = flat
            break

    if exc_dir is None:
        msg = (
            f"ParaView output not found for excitation {excitation}. "
            "Ensure the simulation was run with save_step >= 1."
        )
        raise FileNotFoundError(msg)

    if cycle is not None:
        pvtu_dir = exc_dir / f"Cycle{cycle:06d}"
        candidates = sorted(pvtu_dir.rglob("*.pvtu"))
        if not candidates:
            msg = f"No .pvtu files found in {pvtu_dir}"
            raise FileNotFoundError(msg)
        return candidates[-1]

    # Auto-select last available cycle that contains actual field data.
    # Palace writes a final cycle with only Indicator/Rank (mesh partition);
    # skip it and pick the latest cycle with real solution fields.
    candidates = sorted(exc_dir.rglob("*.pvtu"), reverse=True)
    if not candidates:
        msg = (
            f"No .pvtu files found under {exc_dir}. "
            "Ensure the simulation was run with save_step >= 1."
        )
        raise FileNotFoundError(msg)

    import pyvista as pv

    _partition_only = {"Indicator", "Rank"}
    for pvtu in candidates:
        ds = pv.read(str(pvtu))
        if set(ds.point_data.keys()) != _partition_only:
            return pvtu

    # All cycles are partition-only — return the last one and let the
    # caller surface the "field not found" error with context.
    return candidates[0]
