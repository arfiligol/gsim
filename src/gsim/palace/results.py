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
class EigenmodeReport:
    """Composed Palace Eigenmode report tables for notebook workflows."""

    eigenmodes: Eigenmodes
    mode_history: pd.DataFrame
    pass_summary: pd.DataFrame
    domain_energy: pd.DataFrame
    surface_q: pd.DataFrame
    surface_interface_summary: pd.DataFrame
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

    surface_interface_summary = summarize_surface_q_by_interface(surface_q)
    return EigenmodeReport(
        eigenmodes=eigenmodes,
        mode_history=mode_history,
        pass_summary=pass_summary,
        domain_energy=domain_energy,
        surface_q=surface_q,
        surface_interface_summary=surface_interface_summary,
        port_epr=port_epr,
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


def _optional_int(value: Any) -> int | None:
    if value is None:
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
