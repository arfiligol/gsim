"""General Palace display helper primitives.

The display package provides reusable visualization primitives for typed data
objects: trace figures, trace subplots, heatmaps, and deterministic notebook
display of tables/figures. It does not know about Palace problem reports,
resolve run directories, parse files, or decide which results should be shown.

Typed data classes choose the display primitive that matches their semantics;
problem reports simply aggregate those typed-data visualizers.
"""

from __future__ import annotations

from gsim.palace.display.primitives import (
    DisplayItem,
    DisplayValue,
    PlotlyFigure,
    TraceMapping,
    VisualizationMap,
    VisualizationProvider,
    collect_visualizations,
    display_items,
    make_bar_figure,
    make_heatmap_figure,
    make_trace_figure,
    make_trace_subplot_figure,
)

__all__ = [
    "DisplayItem",
    "DisplayValue",
    "PlotlyFigure",
    "TraceMapping",
    "VisualizationMap",
    "VisualizationProvider",
    "collect_visualizations",
    "display_items",
    "make_bar_figure",
    "make_heatmap_figure",
    "make_trace_figure",
    "make_trace_subplot_figure",
]
