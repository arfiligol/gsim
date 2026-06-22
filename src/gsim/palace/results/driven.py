"""Driven-problem typed data for Palace results.

This module owns S-parameter result objects produced by Palace Driven
simulations. The objects store network data, expose dataframe exports, and
provide their own visualizers. They do not resolve handoff packages or compose
problem reports.
"""

from __future__ import annotations

import json
import logging
import re
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from gsim.palace.display import (
    DisplayValue,
    PlotlyFigure,
    make_trace_subplot_figure,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pandas as pd

type RealArray = NDArray[np.float64]
type ComplexArray = NDArray[np.complex128]
type SParameterSource = str | Path | Mapping[str, str | Path]


class SParam:
    """A single S-parameter entry (complex-valued vs frequency)."""

    def __init__(self, db: RealArray, deg: RealArray) -> None:
        """Create from dB magnitude and degree phase arrays."""
        self._db = db
        self._deg = deg

    @property
    def db(self) -> RealArray:
        """Magnitude in dB."""
        return self._db

    @property
    def deg(self) -> RealArray:
        """Phase in degrees."""
        return self._deg

    @property
    def mag(self) -> RealArray:
        """Linear magnitude."""
        return cast("RealArray", 10 ** (self._db / 20))

    @property
    def complex(self) -> ComplexArray:
        """Complex S-parameter values."""
        return cast("ComplexArray", self.mag * np.exp(1j * np.deg2rad(self._deg)))

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
        freq: RealArray,
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
    def freq(self) -> RealArray:
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

    def to_dataframe(self) -> pd.DataFrame:
        """Export to a flat pandas DataFrame."""
        import pandas as pd

        cols: dict[str, RealArray] = {"freq_ghz": self._freq}
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

    def plot_plotly(self, *, full: bool = False) -> PlotlyFigure:
        """Plot S-parameters with Plotly (interactive).

        Returns a ``plotly.graph_objects.Figure`` that renders interactively
        in notebooks and can be saved as standalone HTML via
        ``fig.write_html("sparams.html")``.
        """
        magnitude_traces: list[dict[str, Any]] = []
        phase_traces: list[dict[str, Any]] = []
        for label, sp in self._filtered_entries(full):
            magnitude_traces.append(
                {
                    "x": self._freq,
                    "y": sp.db,
                    "mode": "lines",
                    "name": label,
                    "legendgroup": label,
                }
            )
            phase_traces.append(
                {
                    "x": self._freq,
                    "y": sp.deg,
                    "mode": "lines",
                    "name": label,
                    "legendgroup": label,
                    "showlegend": False,
                }
            )
        fig = make_trace_subplot_figure(
            (
                {
                    "title": "Magnitude (dB)",
                    "y_title": "dB",
                    "traces": magnitude_traces,
                },
                {
                    "title": "Phase (deg)",
                    "x_title": "Frequency (GHz)",
                    "y_title": "deg",
                    "traces": phase_traces,
                },
            ),
            title="S-Parameters",
        )
        fig.update_layout(height=600)
        return fig

    def figures(self, *, full: bool = False) -> dict[str, PlotlyFigure]:
        """Return visual figures owned by this S-parameter data object."""
        return {"s_parameters_trace_plot": self.plot_plotly(full=full)}

    def tables(self) -> dict[str, pd.DataFrame]:
        """Return tabular views owned by this S-parameter data object."""
        return {"s_parameters_table": self.to_dataframe()}

    def visualize(self, *, full: bool = False) -> dict[str, DisplayValue]:
        """Return all default S-parameter tables and figures."""
        return {**self.tables(), **self.figures(full=full)}

    def _sij_label(self, to_port: str, from_port: str) -> str:
        """Return native ``Sij`` notation for a named port pair."""
        return (
            f"S{self._port_names.index(to_port) + 1}"
            f"{self._port_names.index(from_port) + 1}"
        )

    def plot_interactive(self, phase: bool = False) -> PlotlyFigure:
        """Plot one interactive S-parameter axis using native ``Sij`` labels.

        This preserves gsim's notebook convenience API for manual RF inspection.
        Report visualizers use ``visualize()`` so default reports still show the
        combined magnitude/phase trace plot through the shared display layer.
        """
        import plotly.graph_objects as go  # pyright: ignore[reportMissingTypeStubs]

        first_from = self._port_names[0] if self._port_names else None
        fig = go.Figure()
        for (to_port, from_port), sparam in self._data.items():
            label = self._sij_label(to_port, from_port)
            fig.add_scatter(
                x=self._freq,
                y=sparam.deg if phase else sparam.db,
                mode="lines",
                name=label,
                visible=True if from_port == first_from else "legendonly",
            )
        y_title = "Phase (deg)" if phase else "|S| (dB)"
        fig.update_layout(
            xaxis_title="Frequency (GHz)",
            yaxis_title=y_title,
            title="S-Parameters",
            legend={"groupclick": "toggleitem"},
        )
        return cast("PlotlyFigure", fig)

    def save_npz(self, filepath: str | Path) -> Path:
        """Save S-parameters to a ``.npz`` file.

        This is an export helper for NumPy-based analysis outside the Palace
        run folder. Loading Palace results remains the resolver/loader owner's
        responsibility.

        Args:
            filepath: Destination path (``.npz`` suffix added if missing).

        Returns:
            The resolved file path.
        """
        filepath = Path(filepath).with_suffix(".npz")
        filepath.parent.mkdir(parents=True, exist_ok=True)

        arrays: dict[str, object] = {"freq": self._freq}
        arrays["port_names"] = np.array(self._port_names)
        for (to_p, from_p), sp in self._data.items():
            arrays[f"S_{to_p}_{from_p}_db"] = sp.db
            arrays[f"S_{to_p}_{from_p}_deg"] = sp.deg

        np.savez_compressed(str(filepath), **arrays)  # pyright: ignore[reportArgumentType]
        logger.info("S-parameters saved to %s", filepath)
        return filepath

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


def load_sparams(
    source: SParameterSource,
    *,
    port_info_path: str | Path | None = None,
) -> SParams:
    """Load Palace Driven ``port-S.csv`` results into ``SParams`` typed data."""
    import pandas as pd

    csv_path, base_dir = resolve_sparameter_source(source)
    if csv_path is None:  # pragma: no cover - resolve helper raises first
        msg = "port-S.csv not found"
        raise FileNotFoundError(msg)

    if port_info_path is None and isinstance(source, Mapping):
        port_info_value = source.get("port_information.json")
        if port_info_value is not None:
            port_info_path = Path(port_info_value)

    port_map = _load_port_map(base_dir, csv_path, port_info_path)

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    freq_column = next(
        (column for column in df.columns if column.startswith("f")),
        None,
    )
    freq = cast(
        "RealArray",
        df[freq_column].to_numpy() if freq_column else np.arange(len(df)),
    )

    raw: dict[tuple[int, int], dict[str, RealArray]] = {}
    for column in df.columns:
        parsed = _parse_sparam_column(column)
        if parsed is None:
            continue
        i, j, kind = parsed
        raw.setdefault((i, j), {})[kind] = cast(
            "RealArray",
            df[column].to_numpy(),
        )

    all_indices = set()
    for i, j in raw:
        all_indices.add(i)
        all_indices.add(j)
    port_names = [port_map.get(index, f"p{index}") for index in sorted(all_indices)]

    data: dict[tuple[str, str], SParam] = {}
    for (i, j), parts in sorted(raw.items()):
        to_name = port_map.get(i, f"p{i}")
        from_name = port_map.get(j, f"p{j}")
        db = parts.get("db", cast("RealArray", np.zeros(len(freq))))
        deg = parts.get("deg", cast("RealArray", np.zeros(len(freq))))
        data[(to_name, from_name)] = SParam(db=db, deg=deg)

    files = (
        {str(name): Path(value) for name, value in source.items()}
        if isinstance(source, Mapping)
        else None
    )
    return SParams(freq=freq, data=data, port_names=port_names, files=files)


def resolve_sparameter_source(
    source: SParameterSource,
    *,
    require_csv: bool = True,
) -> tuple[Path | None, Path]:
    """Resolve a Driven S-parameter source into ``(csv_path, base_dir)``."""
    if isinstance(source, Mapping):
        csv_value = source.get("port-S.csv")
        if csv_value is not None:
            csv_path = Path(csv_value)
            return csv_path, csv_path.parent
        for value in source.values():
            path = Path(value)
            if path.exists():
                return None, path.parent
        if require_csv:
            msg = "Results dict has no 'port-S.csv' entry"
            raise FileNotFoundError(msg)
        return None, Path()

    output_dir = Path(source)
    csv_path = _find_sparameter_file(output_dir)
    if csv_path is None and require_csv:
        msg = f"port-S.csv not found in {output_dir} or its subdirectories"
        raise FileNotFoundError(msg)
    return csv_path, output_dir


def _find_sparameter_file(base: Path) -> Path | None:
    candidates = (
        base / "port-S.csv",
        base / "results" / "palace" / "port-S.csv",
        base / "metadata" / "port-S.csv",
        base / "input" / "port-S.csv",
        base / "palace" / "port-S.csv",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _parse_sparam_column(column: str) -> tuple[int, int, str] | None:
    magnitude_match = re.match(r"\|S\[(\d+)\]\[(\d+)\]\|\s*\((\w+\.?)\)", column)
    if magnitude_match:
        return int(magnitude_match.group(1)), int(magnitude_match.group(2)), "db"

    phase_match = re.match(r"arg\(S\[(\d+)\]\[(\d+)\]\)\s*\((\w+\.?)\)", column)
    if phase_match:
        return int(phase_match.group(1)), int(phase_match.group(2)), "deg"

    return None


def _load_port_map(
    output_dir: Path,
    csv_path: Path | None,
    port_info_path: str | Path | None = None,
) -> dict[int, str]:
    if port_info_path is not None:
        info_path = Path(port_info_path)
    else:
        info_path = _find_port_info(output_dir, csv_path)

    if info_path is None or not info_path.exists():
        logger.warning(
            "port_information.json not found - using numeric port names (p1, p2, ...)"
        )
        return {}

    data = json.loads(info_path.read_text(encoding="utf-8"))

    port_map: dict[int, str] = {}
    for entry in data.get("ports", []):
        port_number = entry.get("portnumber")
        name = entry.get("name")
        if port_number is not None and name is not None:
            port_map[port_number] = name
        elif port_number is not None:
            port_map[port_number] = f"p{port_number}"

    if port_map and all(
        name.startswith("p") and name[1:].isdigit() for name in port_map.values()
    ):
        logger.info(
            "port_information.json has no 'name' fields - "
            "columns will use numeric names (p1, p2, ...). "
            "Re-mesh to get named columns."
        )

    return port_map


def _find_port_info(output_dir: Path, csv_path: Path | None) -> Path | None:
    name = "port_information.json"
    candidates = [
        output_dir / name,
        output_dir / "metadata" / name,
    ]
    if csv_path is not None:
        candidates.insert(0, csv_path.parent / name)
        candidates.append(csv_path.parent.parent / "metadata" / name)
        candidates.append(csv_path.parent.parent.parent / "metadata" / name)
        candidates.append(csv_path.parent.parent / "input" / name)
        candidates.append(csv_path.parent.parent / name)
        candidates.append(csv_path.parent.parent.parent / name)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None
