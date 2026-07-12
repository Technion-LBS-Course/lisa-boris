"""Tests for shared dashboard Folium basemap helpers."""

import pytest

folium = pytest.importorskip("folium")

from src.dashboards import map_tiles


def _children_by_class(map_obj, class_name: str):
    return [
        child for child in map_obj._children.values()
        if child.__class__.__name__ == class_name
    ]


def test_satellite_basemap_is_added_once():
    m = folium.Map(location=[32.0853, 34.7818], zoom_start=7, tiles="OpenStreetMap")

    map_tiles.add_satellite_basemap(m)
    map_tiles.add_satellite_basemap(m)

    satellite_layers = [
        child for child in _children_by_class(m, "TileLayer")
        if getattr(child, "layer_name", None) == map_tiles.SATELLITE_LAYER_NAME
    ]
    assert len(satellite_layers) == 1
    assert map_tiles.ESRI_WORLD_IMAGERY_TILES in satellite_layers[0].tiles
    assert map_tiles.ESRI_WORLD_IMAGERY_ATTRIBUTION in satellite_layers[0].options["attribution"]


def test_layer_control_is_added_once():
    m = folium.Map(location=[32.0853, 34.7818], zoom_start=7, tiles="OpenStreetMap")

    map_tiles.add_layer_control_once(m)
    map_tiles.add_layer_control_once(m)

    assert len(_children_by_class(m, "LayerControl")) == 1
