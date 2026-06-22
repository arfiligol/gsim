"""General notebook visualization primitives for Palace typed data.

This module provides small, reusable helpers for the display categories shared
by Palace typed data:

- table-like objects, which are passed through without interpretation;
- trace plots, for one or more x/y series;
- trace subplot grids, for related trace groups such as magnitude and phase;

The helpers deliberately do not know about Driven, Eigenmode, Electrostatic,
loss budgets, handoff packages, or report presets. Domain-specific choices
belong to typed data objects and problem reports.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol, TypeGuard, cast

type DisplayValue = object
type VisualizationMap = Mapping[str, DisplayValue]
type TraceMapping = Mapping[str, Any]


class PlotlyFigure(Protocol):
    """Plotly-compatible figure returned by Display figure builders.

    The protocol keeps Display's public return type concrete for static review
    without depending on Plotly's dynamic runtime export surface.
    """

    layout: Any

    def update_layout(self, *args: Any, **kwargs: Any) -> PlotlyFigure:
        """Update figure layout."""
        ...

    def update_xaxes(self, *args: Any, **kwargs: Any) -> PlotlyFigure:
        """Update x-axis layout."""
        ...

    def update_yaxes(self, *args: Any, **kwargs: Any) -> PlotlyFigure:
        """Update y-axis layout."""
        ...

    def write_html(self, *args: Any, **kwargs: Any) -> Any:
        """Write the figure to HTML using Plotly's runtime implementation."""
        ...


class VisualizationProvider(Protocol):
    """Object that owns a semantic visualization surface."""

    def visualize(self) -> VisualizationMap:
        """Return named tables, figures, or metrics for this object."""
        ...


def collect_visualizations(
    providers: Iterable[VisualizationProvider | None],
) -> dict[str, DisplayValue]:
    """Collect visualizer outputs while rejecting duplicate display keys.

    Display keys carry the semantic presentation contract for a report. A
    duplicate key usually means two typed data objects are trying to show the
    same table or plot, so fail early instead of silently overwriting one view.
    """
    items: dict[str, DisplayValue] = {}
    for provider in providers:
        if provider is None:
            continue
        for name, value in provider.visualize().items():
            if name in items:
                msg = f"Duplicate Palace display item {name!r}"
                raise ValueError(msg)
            items[name] = value
    return items


def make_trace_figure(
    traces: Iterable[TraceMapping],
    *,
    title: str | None = None,
    x_title: str | None = None,
    y_title: str | None = None,
) -> PlotlyFigure:
    """Build a Plotly line/scatter figure from generic trace mappings.

    Each trace mapping may contain ``x``, ``y``, ``name``, ``mode``, and any
    additional Plotly scatter keyword.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    for trace in traces:
        payload = dict(trace)
        payload.setdefault("mode", "lines")
        fig.add_scatter(**payload)
    fig.update_layout(title=title)
    if x_title is not None:
        fig.update_xaxes(title_text=x_title)
    if y_title is not None:
        fig.update_yaxes(title_text=y_title)
    return cast("PlotlyFigure", fig)


def make_bar_figure(
    bars: Iterable[TraceMapping],
    *,
    title: str | None = None,
    x_title: str | None = None,
    y_title: str | None = None,
    barmode: str | None = None,
) -> PlotlyFigure:
    """Build a Plotly bar figure from generic bar trace mappings."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for bar in bars:
        fig.add_bar(**dict(bar))
    fig.update_layout(title=title)
    if barmode is not None:
        fig.update_layout(barmode=barmode)
    if x_title is not None:
        fig.update_xaxes(title_text=x_title)
    if y_title is not None:
        fig.update_yaxes(title_text=y_title)
    return cast("PlotlyFigure", fig)


def make_trace_subplot_figure(
    panels: Sequence[TraceMapping],
    *,
    title: str | None = None,
    shared_xaxes: bool = True,
    vertical_spacing: float = 0.08,
) -> PlotlyFigure:
    """Build a vertical Plotly subplot figure from generic trace panels.

    Each panel mapping must contain ``traces`` and may contain ``title``,
    ``x_title``, and ``y_title``.
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=len(panels),
        cols=1,
        shared_xaxes=shared_xaxes,
        vertical_spacing=vertical_spacing,
        subplot_titles=tuple(str(panel.get("title", "")) for panel in panels),
    )
    shared_x_title: str | None = None
    for row, panel in enumerate(panels, start=1):
        for trace in panel.get("traces", ()):
            payload = dict(trace)
            payload.setdefault("mode", "lines")
            fig.add_scatter(**payload, row=row, col=1)
        if panel.get("x_title") is not None:
            if shared_xaxes:
                shared_x_title = str(panel["x_title"])
            else:
                fig.update_xaxes(title_text=str(panel["x_title"]), row=row, col=1)
        if panel.get("y_title") is not None:
            fig.update_yaxes(title_text=str(panel["y_title"]), row=row, col=1)
    if shared_x_title is not None:
        fig.update_xaxes(title_text=shared_x_title, row=len(panels), col=1)
    fig.update_layout(title=title, height=max(450, 300 * len(panels)))
    return cast("PlotlyFigure", fig)


def display_items(
    items: Mapping[str, DisplayValue],
    *,
    figure_layout: Mapping[str, Any] | None = None,
    plotly_config: Mapping[str, Any] | None = None,
) -> None:
    """Display named tables, figures, and metrics in deterministic order.

    Typed Data plot builders create Plotly figures with semantic data and
    semantic layout: traces, titles, axes, and units. This display helper may
    apply cosmetic Plotly layout such as template, height, margins, or fonts.
    ``figure_layout`` is applied with ``fig.update_layout(...)`` and changes
    cosmetic figure layout. ``plotly_config`` is passed to ``plotly.io.show``
    at render time for options such as responsiveness and modebar controls.
    Plotly config is not stored on the figure object; it is consumed when the
    figure is shown.
    """
    from IPython.display import Markdown, display

    for name, value in items.items():
        if _is_plotly_figure(value):
            if figure_layout is not None:
                value.update_layout(**dict(figure_layout))
            if plotly_config is not None:
                import plotly.io as pio

                pio.show(value, config=dict(plotly_config))
            else:
                display(value)
            continue
        display(Markdown(f"#### {name.replace('_', ' ').title()}"))
        display(_display_value_for_name(name, value))


def _is_plotly_figure(value: DisplayValue) -> TypeGuard[PlotlyFigure]:
    return hasattr(value, "update_layout") and hasattr(value, "write_html")


def _display_value_for_name(name: str, value: DisplayValue) -> DisplayValue:
    if name not in {"domain_epr_summary_table", "surface_epr_summary_table"}:
        return value

    try:
        import pandas as pd
    except ImportError:
        return value

    if not isinstance(value, pd.DataFrame):
        return value

    formats = {
        column: "{:.3e}"
        for column in ("participation", "inverse_q", "q_equivalent", "t1_us")
        if column in value.columns
    }
    return value.style.format(formats) if formats else value


__all__ = [
    "DisplayValue",
    "PlotlyFigure",
    "TraceMapping",
    "VisualizationMap",
    "VisualizationProvider",
    "collect_visualizations",
    "display_items",
    "make_bar_figure",
    "make_trace_figure",
    "make_trace_subplot_figure",
]
