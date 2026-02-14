"""
Unit and integration tests for kml2ofds package.
"""

import json
import os
import sys
from pathlib import Path

import pytest

# Add src to path for imports when running tests directly
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


class TestParseSpanEndpoint:
    """Tests for geometry.parse_span_endpoint."""

    def test_parse_dict(self):
        from kml2ofds.geometry import parse_span_endpoint

        d = {"id": "abc", "name": "Foo"}
        assert parse_span_endpoint(d) == d

    def test_parse_json_string(self):
        from kml2ofds.geometry import parse_span_endpoint

        d = {"id": "abc", "name": "Foo"}
        s = json.dumps(d)
        assert parse_span_endpoint(s) == d

    def test_parse_none(self):
        from kml2ofds.geometry import parse_span_endpoint

        assert parse_span_endpoint(None) is None


class TestRemoveDuplicateNodes:
    """Tests for kml_parse.remove_duplicate_nodes."""

    def test_removes_duplicates(self):
        from kml2ofds.kml_parse import remove_duplicate_nodes

        nodes = [
            {
                "properties": {"name": "A"},
                "geometry": {"coordinates": [1.0, 2.0]},
            },
            {
                "properties": {"name": "A"},
                "geometry": {"coordinates": [1.0, 2.0]},
            },
            {
                "properties": {"name": "B"},
                "geometry": {"coordinates": [3.0, 4.0]},
            },
        ]
        result = remove_duplicate_nodes(nodes, precision=1)
        assert len(result) == 2

    def test_preserves_unique(self):
        from kml2ofds.kml_parse import remove_duplicate_nodes

        nodes = [
            {
                "properties": {"name": "A"},
                "geometry": {"coordinates": [1.0, 2.0]},
            },
            {
                "properties": {"name": "B"},
                "geometry": {"coordinates": [3.0, 4.0]},
            },
        ]
        result = remove_duplicate_nodes(nodes, precision=1)
        assert len(result) == 2


class TestIsPlacemarkIgnored:
    """Tests for _is_placemark_ignored."""

    def test_matches_pattern(self):
        from kml2ofds.kml_parse import _is_placemark_ignored

        assert _is_placemark_ignored("IgnoreThis", ["Ignore"]) is True
        assert _is_placemark_ignored("Foo", ["Ignore"]) is False

    def test_empty_patterns(self):
        from kml2ofds.kml_parse import _is_placemark_ignored

        assert _is_placemark_ignored("Anything", []) is False


class TestFindEndPoint:
    """Tests for geometry.find_end_point."""

    def test_finds_matching_node(self):
        import geopandas as gpd
        from shapely.geometry import Point

        from kml2ofds.geometry import find_end_point

        nodes = gpd.GeoDataFrame(
            {
                "id": ["n1"],
                "name": ["Node 1"],
                "geometry": [Point(0, 0)],
            }
        )
        result = find_end_point((0, 0), nodes, tolerance=1e-2)
        assert result is not None
        assert result["id"] == "n1"

    def test_returns_none_when_no_match(self):
        import geopandas as gpd
        from shapely.geometry import Point

        from kml2ofds.geometry import find_end_point

        nodes = gpd.GeoDataFrame(
            {
                "id": ["n1"],
                "name": ["Node 1"],
                "geometry": [Point(100, 100)],
            }
        )
        result = find_end_point((0, 0), nodes, tolerance=1e-2)
        assert result is None


class TestConfig:
    """Tests for config loading."""

    def test_load_config(self):
        from kml2ofds.config import load_config

        cfg_path = ROOT / "tests" / "fixtures" / "minimal.ini"
        if not cfg_path.exists():
            pytest.skip("fixture not found")
        config = load_config(str(cfg_path))
        assert config.kml_file_name == "minimal.kml"
        assert config.network_name == "Test Network"
        assert config.input_directory == "tests/fixtures/"


class TestIntegration:
    """Integration test: full pipeline on minimal KML."""

    def test_full_pipeline(self):
        """Run full pipeline on minimal.kml and verify outputs."""
        from kml2ofds.config import load_config
        from kml2ofds.cli import _run_pipeline, _ensure_directories


        cfg_path = ROOT / "tests" / "fixtures" / "minimal.ini"
        if not cfg_path.exists():
            pytest.skip("fixture not found")

        # Change to project root so paths resolve
        os.chdir(ROOT)
        config = load_config(str(cfg_path))
        _ensure_directories(config)

        # Ensure KML exists
        kml_path = config.kml_path()
        if not os.path.exists(kml_path):
            pytest.skip(f"KML not found: {kml_path}")

        _run_pipeline(config)

        paths = config.output_paths()
        assert os.path.exists(paths.nodes_geojson)
        assert os.path.exists(paths.spans_geojson)
        assert os.path.exists(paths.ofds_json)

        with open(paths.nodes_geojson) as f:
            nodes = json.load(f)
        with open(paths.spans_geojson) as f:
            spans = json.load(f)

        assert "features" in nodes
        assert "features" in spans
        assert len(nodes["features"]) >= 2
        assert len(spans["features"]) >= 1
