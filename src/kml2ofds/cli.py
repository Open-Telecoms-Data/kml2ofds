"""
CLI entry point and pipeline orchestration for kml2ofds.
"""

import os
import sys
from importlib.metadata import version

import click

from .config import load_config, Config
from .api import run_pipeline

__version__ = version("kml2ofds")


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


def _print_version(ctx: click.Context, _param: click.Parameter, value: bool) -> None:
    if value:
        click.echo(f"kml2ofds, version {__version__}")
        ctx.exit()


@click.command(help="Convert KML files to the Open Fibre Data Standard format.")
@click.option(
    "-V",
    "--version",
    is_flag=True,
    is_eager=True,
    expose_value=False,
    help="Show the version and exit.",
    callback=_print_version,
)
@click.option(
    "--network-profile",
    required=True,
    help="Path to the network profile configuration file (required).",
    type=click.Path(exists=True),
)
@click.option(
    "--merge-proximate-nodes",
    is_flag=True,
    default=False,
    help="Enable proximate node merge (overrides profile).",
)
@click.option(
    "--no-merge-proximate-nodes",
    is_flag=True,
    default=False,
    help="Disable proximate node merge (overrides profile).",
)
@click.option(
    "--merge-proximate-nodes-meters",
    type=float,
    default=None,
    help="Proximity threshold in metres for node merge (overrides profile).",
)
def main(
    network_profile: str,
    merge_proximate_nodes: bool,
    no_merge_proximate_nodes: bool,
    merge_proximate_nodes_meters: float | None,
) -> None:
    """Convert KML files to the Open Fibre Data Standard format."""
    print(f"Running with network_profile: {network_profile}")
    if merge_proximate_nodes and no_merge_proximate_nodes:
        raise click.UsageError(
            "Use only one of --merge-proximate-nodes and "
            "--no-merge-proximate-nodes."
        )
    config = load_config(network_profile)
    if merge_proximate_nodes:
        config.merge_proximate_nodes = True
    elif no_merge_proximate_nodes:
        config.merge_proximate_nodes = False
    if merge_proximate_nodes_meters is not None:
        config.merge_proximate_nodes_meters = float(
            merge_proximate_nodes_meters
        )
    _ensure_directories(config)
    _validate_kml_exists(config)
    run_pipeline(config)
