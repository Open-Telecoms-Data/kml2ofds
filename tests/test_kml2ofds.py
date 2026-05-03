"""
Unit and integration tests for kml2ofds package.
"""

import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import pytest
from click.testing import CliRunner

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


class TestBreakSpansAtNodePoints:
    """Regression: STRtree bbox miss for near-collinear snapped nodes."""

    def test_splits_when_node_slightly_off_axis_aligned_line(self):
        """Snapped point with tiny y residual must still break horizontal span."""
        import geopandas as gpd
        from shapely.geometry import LineString, Point

        from kml2ofds.geometry import break_spans_at_node_points

        line = LineString([(0.0, 0.0), (10.0, 0.0)])
        # Bbox of line is flat on y=0; point y>0 does not overlap line bbox with
        # STRtree.query(line), but the point is still within NODE_SNAP of the line.
        node = Point(5.0, 1e-12)
        gdf_spans = gpd.GeoDataFrame(
            {"name": ["main"], "geometry": [line]},
            crs="EPSG:4326",
        )
        gdf_nodes = gpd.GeoDataFrame(
            {"geometry": [node]},
            crs="EPSG:4326",
        )
        out = break_spans_at_node_points(
            gdf_nodes,
            gdf_spans,
            "N",
            "nid",
            [],
            buffer_size=1e-4,
        )
        # Must split (not stay a single span); buffer/split may yield 2+ pieces.
        assert len(out) >= 2

    def test_merge_after_break_rejoins_node_splits(self):
        """merge after break rejoins splits; pipeline runs merge first."""
        import geopandas as gpd
        from shapely.geometry import LineString, Point

        from kml2ofds.geometry import (
            break_spans_at_node_points,
            merge_contiguous_spans,
        )

        line = LineString([(0, 0), (1, 0.05), (2, 0)])
        node = Point(1.0, 0.05)
        gdf_spans = gpd.GeoDataFrame(
            {"name": ["s"], "geometry": [line]},
            crs="EPSG:4326",
        )
        gdf_nodes = gpd.GeoDataFrame({"geometry": [node]}, crs="EPSG:4326")
        broken = break_spans_at_node_points(
            gdf_nodes,
            gdf_spans,
            "N",
            "nid",
            [],
            buffer_size=1e-4,
        )
        assert len(broken) >= 2
        rejoined = merge_contiguous_spans(broken, precision=6)
        assert len(rejoined) == 1


class TestMergeProximateNodes:
    """merge_proximate_nodes clusters and naming."""

    def test_merges_two_close_nodes_combines_names(self):
        import geopandas as gpd
        from shapely.geometry import Point

        from kml2ofds.geometry import merge_proximate_nodes

        # ~11 m east of (0,0) at the equator — within 50 m threshold
        gdf = gpd.GeoDataFrame(
            {
                "id": ["a", "b"],
                "name": ["Node A", "Node B"],
                "geometry": [Point(0, 0), Point(1e-4, 0)],
            },
            crs="EPSG:4326",
        )
        out = merge_proximate_nodes(gdf, 50.0)
        assert len(out) == 1
        assert out.iloc[0]["name"] == "Node A / Node B"

    def test_singleton_unchanged(self):
        import geopandas as gpd
        from shapely.geometry import Point

        from kml2ofds.geometry import merge_proximate_nodes

        gdf = gpd.GeoDataFrame(
            {"id": ["x"], "name": ["Only"], "geometry": [Point(1, 1)]},
            crs="EPSG:4326",
        )
        out = merge_proximate_nodes(gdf, 50.0)
        assert len(out) == 1
        assert out.iloc[0]["name"] == "Only"


class TestMergeContiguousSpans:
    """merge_contiguous_spans node-at-junction behaviour."""

    def test_does_not_merge_two_spans_when_node_at_shared_endpoint(self):
        """Separate KML paths meeting at a tip stay separate if a node marks it."""
        import geopandas as gpd
        from shapely.geometry import LineString, Point

        from kml2ofds.geometry import merge_contiguous_spans

        tip = (1.0, 0.1)
        span1 = LineString([(0, 0), tip])
        span2 = LineString([tip, (2, 0)])
        gdf = gpd.GeoDataFrame(
            {"name": ["arm_a", "arm_b"], "geometry": [span1, span2]},
            crs="EPSG:4326",
        )
        nodes = gpd.GeoDataFrame(
            {"geometry": [Point(tip)]},
            crs="EPSG:4326",
        )
        out = merge_contiguous_spans(
            gdf, precision=6, gdf_nodes=nodes
        )
        assert len(out) == 2

    def test_merges_when_no_nodes_given(self):
        import geopandas as gpd
        from shapely.geometry import LineString

        from kml2ofds.geometry import merge_contiguous_spans

        tip = (1.0, 0.1)
        span1 = LineString([(0, 0), tip])
        span2 = LineString([tip, (2, 0)])
        gdf = gpd.GeoDataFrame(
            {"name": ["arm_a", "arm_b"], "geometry": [span1, span2]},
            crs="EPSG:4326",
        )
        out = merge_contiguous_spans(gdf, precision=6, gdf_nodes=None)
        assert len(out) == 1


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

    def test_config_from_dict_generates_provider_uuids_when_blank(self):
        import uuid

        from kml2ofds.config import config_from_dict

        c = config_from_dict({"kml_file_name": "x.kml"})
        uuid.UUID(c.physical_infrastructure_provider_id)
        uuid.UUID(c.network_providers_id)
        assert c.physical_infrastructure_provider_id != c.network_providers_id

    def test_postprocess_to_ofds_04_defaults_true(self):
        from kml2ofds.config import config_from_dict

        cfg = config_from_dict({"kml_file_name": "x.kml"})
        assert cfg.postprocess_to_ofds_04 is True

    def test_postprocess_to_ofds_04_can_be_disabled(self):
        from kml2ofds.config import config_from_dict

        cfg = config_from_dict(
            {
                "kml_file_name": "x.kml",
                "postprocess_to_ofds_04": "false",
            }
        )
        assert cfg.postprocess_to_ofds_04 is False


class TestOfdsMigration:
    """Tests for post-export OFDS 0.3->0.4 migration."""

    def test_migrate_package_moves_legacy_fields(self):
        from kml2ofds.ofds_migrate import (
            OFDS_03_SCHEMA_URL,
            OFDS_04_SCHEMA_URL,
            migrate_package_03_to_04,
        )

        source = {
            "networks": [
                {
                    "id": "n",
                    "links": [{"rel": "describedby", "href": OFDS_03_SCHEMA_URL}],
                    "nodes": [
                        {
                            "id": "node-1",
                            "physicalInfrastructureProvider": {"id": "org-1"},
                        }
                    ],
                    "spans": [
                        {
                            "id": "span-1",
                            "physicalInfrastructureProvider": {"id": "org-2"},
                            "deploymentDetails": {"description": "legacy detail"},
                        }
                    ],
                }
            ]
        }

        migrated, result = migrate_package_03_to_04(source)
        network = migrated["networks"][0]
        node = network["nodes"][0]
        span = network["spans"][0]

        assert network["links"][0]["href"] == OFDS_04_SCHEMA_URL
        assert "physicalInfrastructureProvider" not in node
        assert node["transmissionMediumOwner"]["id"] == "org-1"
        assert "physicalInfrastructureProvider" not in span
        assert span["transmissionMediumOwner"]["id"] == "org-2"
        assert "deploymentDetails" not in span
        assert span["supportingInfrastructure"]["description"] == "legacy detail"
        assert result.schema_links_updated == 1
        assert result.node_provider_fields_migrated == 1
        assert result.span_provider_fields_migrated == 1
        assert result.span_deployment_details_migrated == 1

    def test_cli_migrates_file(self, tmp_path):
        from kml2ofds.ofds_migrate import OFDS_03_SCHEMA_URL, OFDS_04_SCHEMA_URL
        from kml2ofds.ofds_migrate_cli import main

        source = {
            "networks": [
                {
                    "id": "net-1",
                    "links": [
                        {"rel": "describedby", "href": OFDS_03_SCHEMA_URL}
                    ],
                    "nodes": [
                        {
                            "id": "node-1",
                            "physicalInfrastructureProvider": {"id": "org-1"},
                        }
                    ],
                    "spans": [
                        {
                            "id": "span-1",
                            "physicalInfrastructureProvider": {"id": "org-2"},
                        }
                    ],
                }
            ]
        }
        input_path = tmp_path / "in.json"
        output_path = tmp_path / "out.json"
        input_path.write_text(json.dumps(source), encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--no-validate",
            ],
        )

        assert result.exit_code == 0
        migrated = json.loads(output_path.read_text(encoding="utf-8"))
        network = migrated["networks"][0]
        assert network["links"][0]["href"] == OFDS_04_SCHEMA_URL
        assert "physicalInfrastructureProvider" not in network["nodes"][0]
        assert "transmissionMediumOwner" in network["nodes"][0]

    def test_json_to_geojson_converter_scales_linearly_with_patch(self):
        """libcoveofds span endpoint resolution is O(spans*nodes) upstream; patch fixes."""
        from libcoveofds.geojson import JSONToGeoJSONConverter

        from kml2ofds.libcoveofds_span_endpoint_patch import (
            apply_libcoveofds_span_endpoint_patch,
        )

        apply_libcoveofds_span_endpoint_patch()
        n = 6000
        nodes = [
            {"id": f"node-{i}", "name": {"lang": "en", "name": f"N{i}"}}
            for i in range(n)
        ]
        spans = [
            {"id": f"span-{i}", "start": f"node-{i}", "end": f"node-{i + 1}"}
            for i in range(n - 1)
        ]
        pkg = {"networks": [{"id": "net", "nodes": nodes, "spans": spans}]}
        t0 = time.monotonic()
        converter = JSONToGeoJSONConverter()
        converter.process_package(deepcopy(pkg))
        elapsed = time.monotonic() - t0
        assert elapsed < 15.0, f"converter took {elapsed:.1f}s (patch missing?)"
        feats = converter.get_spans_geojson()["features"]
        assert len(feats) == n - 1
        start0 = feats[0]["properties"]["start"]
        assert isinstance(start0, dict) and start0.get("id") == "node-0"


class TestIntegration:
    """Integration test: full pipeline on minimal KML."""

    def test_full_pipeline(self):
        """Run full pipeline on minimal.kml and verify outputs."""
        from kml2ofds.config import load_config
        from kml2ofds.api import run_pipeline
        from kml2ofds.cli import _ensure_directories

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

        run_pipeline(config)

        paths = config.output_paths()
        assert os.path.exists(paths.nodes_geojson)
        assert os.path.exists(paths.spans_geojson)
        assert os.path.exists(paths.ofds_json)

        with open(paths.nodes_geojson) as f:
            nodes = json.load(f)
        with open(paths.spans_geojson) as f:
            spans = json.load(f)
        with open(paths.ofds_json) as f:
            ofds = json.load(f)

        assert "features" in nodes
        assert "features" in spans
        assert len(nodes["features"]) >= 2
        assert len(spans["features"]) >= 1
        network = ofds["networks"][0]
        assert (
            network["links"][0]["href"]
            == "https://standard.ofds.info/en/0__4__0/network-schema.json"
        )
        assert len(network.get("nodes", [])) >= 2
        assert len(network.get("spans", [])) >= 1
        assert all(
            "physicalInfrastructureProvider" not in node
            for node in network.get("nodes", [])
        )
        assert all(
            "physicalInfrastructureProvider" not in span
            for span in network.get("spans", [])
        )


class TestRfc4122NetworkId:
    """Tests for optional Network ID validation (RFC 4122 UUID)."""

    def test_empty_accepted(self):
        from kml2ofds.rfc4122 import network_id_validation_error

        assert network_id_validation_error(None) is None
        assert network_id_validation_error("") is None
        assert network_id_validation_error("   ") is None

    def test_valid_uuid_accepted(self):
        from kml2ofds.rfc4122 import network_id_validation_error

        assert (
            network_id_validation_error("550e8400-e29b-41d4-a716-446655440000")
            is None
        )
        assert (
            network_id_validation_error("{550e8400-e29b-41d4-a716-446655440000}")
            is None
        )
        assert (
            network_id_validation_error(
                "urn:uuid:550e8400-e29b-41d4-a716-446655440000"
            )
            is None
        )

    def test_invalid_rejected(self):
        from kml2ofds.rfc4122 import (
            NETWORK_ID_INVALID_MESSAGE,
            network_id_validation_error,
        )

        assert network_id_validation_error("not-a-uuid") == NETWORK_ID_INVALID_MESSAGE
        assert network_id_validation_error("550e8400-e29b-41d4-a716") == (
            NETWORK_ID_INVALID_MESSAGE
        )
        # Nil / all-zero UUID does not use the RFC 4122 variant in Python's model
        assert network_id_validation_error(
            "00000000-0000-0000-0000-000000000000"
        ) == NETWORK_ID_INVALID_MESSAGE
