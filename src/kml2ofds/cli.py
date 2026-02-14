"""
CLI entry point and pipeline orchestration for kml2ofds.
"""

import os
import sys
import click

from .config import load_config, Config

TOTAL_STAGES = 8


def _stage(n: int, msg: str) -> None:
    """Print a stage label."""
    print(f"\n[{n}/{TOTAL_STAGES}] {msg}", flush=True)


def _run_pipeline(config: Config) -> None:
    """Run the full KML-to-OFDS pipeline."""
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
    _stage(1, "Parsing KML...")
    gdf_ofds_nodes, gdf_spans = process_kml_file(config.kml_path(), config)

    # 2. Break spans at node points
    _stage(2, "Breaking spans at node points...")
    min_vert = pd.Series([len(x.coords) for x in gdf_spans.geometry]).min()
    print(
        f"  Before: {len(gdf_spans)} spans, "
        f"{len(gdf_ofds_nodes)} nodes, min vertices: {min_vert}"
    )

    gdf_nodes_for_breaking = filter_ignored_nodes(
        gdf_ofds_nodes, config.ignore_placemarks
    )
    print(
        f"  Filtered out ignored nodes: {len(gdf_ofds_nodes)} -> "
        f"{len(gdf_nodes_for_breaking)} nodes"
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
        f"  After: {len(gdf_spans)} spans, "
        f"{len(gdf_ofds_nodes)} nodes, min vertices: {min_vert}"
    )

    # 3. Optional merge contiguous spans
    if config.merge_contiguous_spans:
        _stage(3, "Merging contiguous spans...")
        spans_before = len(gdf_spans)
        print(
            f"  Precision: {config.merge_contiguous_spans_precision} decimal places"
        )
        print(f"  Before merge: {spans_before} spans")
        gdf_spans = merge_contiguous_spans(
            gdf_spans, precision=config.merge_contiguous_spans_precision
        )
        print(f"  After merge: {len(gdf_spans)} spans ({spans_before - len(gdf_spans)} merged)")
    else:
        _stage(3, "Merging contiguous spans... (skipped)")

    # 4. Add missing nodes
    _stage(4, "Adding missing nodes...")
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
    _stage(5, "Adding nodes to spans...")
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
    _stage(6, "Consolidating auto-generated nodes...")
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
    _stage(7, "Final filter...")
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
    _stage(8, "Exporting...")
    print("  Converting to OFDS JSON (this may take a moment)...", flush=True)
    export_ofds(
        gdf_ofds_nodes,
        gdf_ofds_spans,
        paths,
        validate=config.validate_output,
    )

    print("\nComplete")


def _ensure_directories(config: Config) -> None:
    """Create input, output, and debug directories if needed."""
    if not os.path.exists(config.input_directory):
        os.makedirs(config.input_directory)
    if not os.path.exists(config.output_directory):
        os.makedirs(config.output_directory)
    if config.debug_enabled and not os.path.exists(config.debug_output_directory):
        os.makedirs(config.debug_output_directory)
        print(f"Created debug output directory: {config.debug_output_directory}")


def _validate_kml_exists(config: Config) -> None:
    """Verify KML file exists; exit with helpful message if not."""
    kml_path = config.kml_path()
    directory = os.path.join(os.getcwd(), config.input_directory)

    if not os.path.exists(kml_path):
        print(f"\nERROR: KML file not found!")
        print(f"  Expected file: {kml_path}")
        print(f"  Profile setting: kml_file_name = {config.kml_file_name}")
        print(f"  Input directory: {directory}")
        print(f"  Current working directory: {os.getcwd()}")
        if not os.path.exists(directory):
            print(f"  NOTE: Input directory does not exist: {directory}")
        else:
            print(f"  NOTE: Input directory exists but file not found")
            try:
                files_in_dir = os.listdir(directory)
                if files_in_dir:
                    print(f"  Files in input directory:")
                    for f in sorted(files_in_dir)[:10]:
                        print(f"    - {f}")
                    if len(files_in_dir) > 10:
                        print(f"    ... and {len(files_in_dir) - 10} more files")
                else:
                    print(f"  Input directory is empty")
            except PermissionError:
                print(f"  Could not list files (permission denied)")
        sys.exit(1)


@click.command(help="Convert KML files to the Open Fibre Data Standard format.")
@click.option(
    "--network-profile",
    required=True,
    help="Path to the network profile configuration file (required).",
    type=click.Path(exists=True),
)
def main(network_profile: str) -> None:
    """Convert KML files to the Open Fibre Data Standard format."""
    print(f"Running with network_profile: {network_profile}")
    config = load_config(network_profile)
    _ensure_directories(config)
    _validate_kml_exists(config)
    _run_pipeline(config)
