"""
OFDS export: write GeoJSON and OFDS JSON via libcoveofds.
"""

import json
from typing import TYPE_CHECKING

import geopandas as gpd

if TYPE_CHECKING:
    from .config import OutputPaths


def export_ofds(
    gdf_nodes: gpd.GeoDataFrame,
    gdf_spans: gpd.GeoDataFrame,
    paths: "OutputPaths",
    validate: bool = False,
    postprocess_to_ofds_04: bool = True,
) -> None:
    """Write nodes and spans GeoJSON, convert to OFDS JSON, optionally validate."""
    # Use to_json() instead of to_file(): GDAL/OGR (used by to_file) mishandles nested
    # structures (status, physicalInfrastructureProvider, networkProviders), while
    # to_json() preserves them correctly.
    with open(paths.spans_geojson, "w", encoding="utf-8") as f:
        f.write(gdf_spans.to_json())
    with open(paths.nodes_geojson, "w", encoding="utf-8") as f:
        f.write(gdf_nodes.to_json())
    print("  GeoJSON files written.", flush=True)

    from libcoveofds.geojson import GeoJSONToJSONConverter, GeoJSONAssumeFeatureType

    with open(paths.spans_geojson, encoding="utf-8") as f:
        spans_geojson = json.load(f)
    with open(paths.nodes_geojson, encoding="utf-8") as f:
        nodes_geojson = json.load(f)

    converter = GeoJSONToJSONConverter()
    converter.process_data(
        nodes_geojson, assumed_feature_type=GeoJSONAssumeFeatureType.NODE
    )
    converter.process_data(
        spans_geojson, assumed_feature_type=GeoJSONAssumeFeatureType.SPAN
    )
    ofds_json = converter.get_json()

    with open(paths.ofds_json, "w", encoding="utf-8") as f:
        json.dump(ofds_json, f, indent=4, ensure_ascii=False)

    if postprocess_to_ofds_04:
        from .ofds_migrate import migrate_output_files_to_04

        migration_result = migrate_output_files_to_04(paths, validate=validate)
        print(
            "  Migrated outputs to OFDS 0.4 "
            f"(links: {migration_result.schema_links_updated}, "
            f"nodes: {migration_result.node_provider_fields_migrated}, "
            f"spans: {migration_result.span_provider_fields_migrated}, "
            f"deployment details: {migration_result.span_deployment_details_migrated}).",
            flush=True,
        )

    if validate:
        from libcoveofds.schema import OFDSSchema
        from libcoveofds.python_validate import PythonValidate

        schema = OFDSSchema()
        validator = PythonValidate(schema)
        with open(paths.ofds_json, encoding="utf-8") as f:
            output_for_validation = json.load(f)
        result = validator.validate(output_for_validation)
        if not result:
            print("OFDS validation: passed")
        else:
            print("OFDS validation: failed")
            for err in result[:10]:
                print(f"  {err}")
            if len(result) > 10:
                print(f"  ... and {len(result) - 10} more errors")
