"""
Programmatic API for kml2ofds: run conversion from KML bytes and config dict.
"""

import os
import tempfile
from collections.abc import Callable
from typing import Optional

from .config import Config, config_from_dict, OutputPaths

TOTAL_STAGES = 8


def _stage(
    n: int,
    msg: str,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> None:
    """Print a stage label and optionally invoke progress callback."""
    if progress_callback:
        progress_callback(n, TOTAL_STAGES, msg)
    print(f"\n[{n}/{TOTAL_STAGES}] {msg}", flush=True)


def run_pipeline(
    config: Config,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> None:
    """Run the full KML-to-OFDS pipeline. Writes output to config.output_paths()."""
    print(f"Running pipeline for: {config.kml_file_name}")
    if config.debug_enabled:
        print("Debug mode: enabled")
    print("Loading modules...", flush=True)

    print("  kml_parse...", flush=True)
    from . import kml_parse
    process_kml_file = kml_parse.process_kml_file

    print("  geometry...", flush=True)
    from .geometry import (
        break_spans_at_node_points,
        add_missing_nodes,
        add_nodes_to_spans,
        filter_ignored_nodes,
        merge_contiguous_spans,
        apply_network_status,
        apply_provider_info,
    )

    print("  consolidate...", flush=True)
    from .consolidate import consolidate_auto_generated_nodes

    print("  ofds_export...", flush=True)
    from .ofds_export import export_ofds

    print("  pandas...", flush=True)
    import pandas as pd

    print("Modules loaded.", flush=True)
    paths = config.output_paths()

    # 1. Parse KML
    _stage(1, "Parsing KML...", progress_callback)
    gdf_ofds_nodes, gdf_spans = process_kml_file(config.kml_path(), config)

    # 2. Optional merge contiguous spans (before breaking at nodes).
    # merge_contiguous_spans joins pieces that share endpoints. After
    # break_spans_at_node_points those pieces are head-to-tail; merging then
    # reassembles the line and drops the split (sharp bends, T-junctions, etc.).
    min_vert = pd.Series([len(x.coords) for x in gdf_spans.geometry]).min()
    print(
        f"  After parse: {len(gdf_spans)} spans, "
        f"{len(gdf_ofds_nodes)} nodes, min vertices: {min_vert}"
    )
    if config.merge_contiguous_spans:
        _stage(2, "Merging contiguous spans...", progress_callback)
        spans_before = len(gdf_spans)
        print(
            f"  Precision: {config.merge_contiguous_spans_precision} decimal places"
        )
        print(f"  Before merge: {spans_before} spans")
        gdf_spans = merge_contiguous_spans(
            gdf_spans,
            precision=config.merge_contiguous_spans_precision,
            gdf_nodes=gdf_ofds_nodes,
        )
        merged_n = spans_before - len(gdf_spans)
        print(f"  After merge: {len(gdf_spans)} spans ({merged_n} merged)")
    else:
        _stage(2, "Merging contiguous spans... (skipped)", progress_callback)

    # 3. Break spans at node points
    _stage(3, "Breaking spans at node points...", progress_callback)
    gdf_nodes_for_breaking = filter_ignored_nodes(
        gdf_ofds_nodes, config.ignore_placemarks
    )
    print(
        f"  Filtered out ignored nodes: {len(gdf_ofds_nodes)} -> "
        f"{len(gdf_nodes_for_breaking)} nodes"
    )
    min_vert = pd.Series([len(x.coords) for x in gdf_spans.geometry]).min()
    print(
        f"  Before break: {len(gdf_spans)} spans, min vertices: {min_vert}"
    )

    gdf_spans = break_spans_at_node_points(
        gdf_nodes_for_breaking,
        gdf_spans,
        config.network_name,
        config.network_id,
        config.network_links,
    )

    min_vert = pd.Series([len(x.coords) for x in gdf_spans.geometry]).min()
    print(
        f"  After break: {len(gdf_spans)} spans, min vertices: {min_vert}"
    )

    # 4. Add missing nodes
    _stage(4, "Adding missing nodes...", progress_callback)
    nodes_before = len(gdf_ofds_nodes)
    gdf_ofds_nodes, gdf_auto_gen_nodes = add_missing_nodes(
        gdf_spans,
        gdf_ofds_nodes,
        config.network_id,
        config.network_name,
        config.network_links,
        config.physical_infrastructure_provider_id,
        config.physical_infrastructure_provider_name,
        config.network_providers_id,
        config.network_providers_name,
    )
    print(
        f"  Added {len(gdf_auto_gen_nodes)} missing nodes. "
        f"Total nodes: {nodes_before} -> {len(gdf_ofds_nodes)}"
    )

    # 5. Add nodes to spans
    _stage(5, "Adding nodes to spans...", progress_callback)
    min_vert = pd.Series([len(x.coords) for x in gdf_spans.geometry]).min()
    print(
        f"  Before: {len(gdf_spans)} spans, "
        f"{len(gdf_ofds_nodes)} nodes, min vertices: {min_vert}"
    )
    gdf_ofds_spans = add_nodes_to_spans(gdf_spans, gdf_ofds_nodes)

    # Filter same start/end
    from .geometry import extract_node_id
    spans_before_filter = len(gdf_ofds_spans)
    start_ids = gdf_ofds_spans["start"].apply(extract_node_id)
    end_ids = gdf_ofds_spans["end"].apply(extract_node_id)
    valid_mask = (start_ids != end_ids) | (start_ids.isna()) | (end_ids.isna())
    gdf_ofds_spans = gdf_ofds_spans[valid_mask].copy()
    spans_removed = spans_before_filter - len(gdf_ofds_spans)
    if spans_removed > 0:
        print(f"  Removed {spans_removed} spans with identical start and end nodes")

    min_vert = pd.Series([len(x.coords) for x in gdf_ofds_spans.geometry]).min()
    print(
        f"  After: {len(gdf_ofds_spans)} spans, "
        f"{len(gdf_ofds_nodes)} nodes, min vertices: {min_vert}"
    )

    # 6. Consolidate auto-generated nodes
    _stage(6, "Consolidating auto-generated nodes...", progress_callback)
    spans_before = len(gdf_ofds_spans)
    nodes_before = len(gdf_ofds_nodes)
    min_vert = pd.Series([len(x.coords) for x in gdf_ofds_spans.geometry]).min()
    print(
        f"  Before: {spans_before} spans, "
        f"{nodes_before} nodes, min vertices: {min_vert}"
    )

    gdf_ofds_spans, gdf_ofds_nodes = consolidate_auto_generated_nodes(
        gdf_ofds_nodes,
        gdf_ofds_spans,
        config.threshold_meters,
        debug_enabled=config.debug_enabled,
        debug_output_dir=config.debug_output_directory,
        debug_output_prefix=paths.debug_prefix,
        rename_spans_from_nodes=config.rename_spans_from_nodes,
    )

    spans_after = len(gdf_ofds_spans)
    nodes_after = len(gdf_ofds_nodes)
    min_vert = pd.Series([len(x.coords) for x in gdf_ofds_spans.geometry]).min()
    print(
        f"  After: {spans_after} spans ({spans_after - spans_before} net change), "
        f"{nodes_after} nodes ({nodes_before - nodes_after} removed), "
        f"min vertices: {min_vert}"
    )

    # 7. Final filter same start/end
    _stage(7, "Final filter...", progress_callback)
    spans_before_final = len(gdf_ofds_spans)
    start_ids = gdf_ofds_spans["start"].apply(extract_node_id)
    end_ids = gdf_ofds_spans["end"].apply(extract_node_id)
    valid_mask = (start_ids != end_ids) | (start_ids.isna()) | (end_ids.isna())
    gdf_ofds_spans = gdf_ofds_spans[valid_mask].copy()
    if spans_before_final - len(gdf_ofds_spans) > 0:
        print(
            f"  Removed {spans_before_final - len(gdf_ofds_spans)} additional spans "
            "with identical start and end nodes"
        )

    # 7.5. Apply network status to all nodes and spans
    gdf_ofds_nodes, gdf_ofds_spans = apply_network_status(
        gdf_ofds_nodes, gdf_ofds_spans, config.network_status
    )

    # 7.6. Apply provider info to all nodes and spans
    gdf_ofds_nodes, gdf_ofds_spans = apply_provider_info(
        gdf_ofds_nodes,
        gdf_ofds_spans,
        config.physical_infrastructure_provider_id,
        config.physical_infrastructure_provider_name,
        config.network_providers_id,
        config.network_providers_name,
    )

    # 8. Export
    _stage(8, "Exporting...", progress_callback)
    print("  Converting to OFDS JSON (this may take a moment)...", flush=True)
    export_ofds(
        gdf_ofds_nodes,
        gdf_ofds_spans,
        paths,
        validate=config.validate_output,
    )

    print("\nComplete")


def run_conversion(
    kml_content: bytes,
    config_dict: dict[str, str],
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> dict[str, bytes]:
    """
    Run KML-to-OFDS conversion from in-memory KML and config dict.

    Args:
        kml_content: Raw KML file bytes.
        config_dict: Configuration dict (form data). Keys match INI profile.
        progress_callback: Optional callback(stage, total, message) for progress.

    Returns:
        Dict with keys being output filenames (from config output_paths, e.g.
        "{prefix}_ofds-nodes_{date}.geojson") mapping to file contents as bytes.

    Raises:
        Exception: Any exception from the pipeline (parse errors, etc.).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        kml_filename = os.path.basename(
            config_dict.get("kml_file_name", "upload.kml")
        ) or "upload.kml"
        if not kml_filename.lower().endswith(".kml"):
            kml_filename = "upload.kml"
        kml_path = os.path.join(tmpdir, kml_filename)
        with open(kml_path, "wb") as f:
            f.write(kml_content)

        config_dict_for_config = dict(config_dict)
        config_dict_for_config["kml_file_name"] = kml_filename
        config = config_from_dict(
            config_dict_for_config,
            input_directory=tmpdir,
            output_directory=tmpdir,
        )

        run_pipeline(config, progress_callback=progress_callback)

        paths = config.output_paths()
        result: dict[str, bytes] = {}
        nodes_filename = os.path.basename(paths.nodes_geojson)
        spans_filename = os.path.basename(paths.spans_geojson)
        ofds_filename = os.path.basename(paths.ofds_json)
        with open(paths.nodes_geojson, "rb") as f:
            result[nodes_filename] = f.read()
        with open(paths.spans_geojson, "rb") as f:
            result[spans_filename] = f.read()
        with open(paths.ofds_json, "rb") as f:
            result[ofds_filename] = f.read()

        return result
