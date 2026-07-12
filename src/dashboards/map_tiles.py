"""Shared Folium basemap helpers for Streamlit dashboards."""

from __future__ import annotations

from typing import Any


ESRI_WORLD_IMAGERY_TILES = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
ESRI_WORLD_IMAGERY_ATTRIBUTION = (
    "Tiles (C) Esri - Source: Esri, Maxar, Earthstar Geographics, "
    "and the GIS User Community"
)

SATELLITE_LAYER_NAME = "Satellite"


def _has_child(map_obj: Any, class_name: str, layer_name: str | None = None) -> bool:
    for child in getattr(map_obj, "_children", {}).values():
        if child.__class__.__name__ != class_name:
            continue
        if layer_name is None or getattr(child, "layer_name", None) == layer_name:
            return True
    return False


def add_satellite_basemap(map_obj: Any) -> Any:
    """Add Esri World Imagery as an optional base layer if it is not present."""
    if _has_child(map_obj, "TileLayer", SATELLITE_LAYER_NAME):
        return map_obj

    import folium

    folium.TileLayer(
        tiles=ESRI_WORLD_IMAGERY_TILES,
        attr=ESRI_WORLD_IMAGERY_ATTRIBUTION,
        name=SATELLITE_LAYER_NAME,
        overlay=False,
        control=True,
        max_zoom=19,
    ).add_to(map_obj)
    return map_obj


def add_layer_control_once(map_obj: Any) -> Any:
    """Add one layer toggle to a Folium map."""
    if _has_child(map_obj, "LayerControl"):
        return map_obj

    import folium

    folium.LayerControl(collapsed=True).add_to(map_obj)
    return map_obj
