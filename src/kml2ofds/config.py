"""
Configuration loading and path resolution for kml2ofds.
"""

import configparser
import os
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from .constants import DEFAULT_NETWORK_LINKS


@dataclass
class OutputPaths:
    """Resolved output file paths."""

    nodes_geojson: str
    spans_geojson: str
    ofds_json: str
    debug_prefix: str


@dataclass
class Config:
    """Run configuration for kml2ofds."""

    kml_file_name: str
    network_id: str
    network_name: str
    network_status: str
    network_links: list[dict]
    physical_infrastructure_provider_id: str
    physical_infrastructure_provider_name: str
    network_providers_id: str
    network_providers_name: str
    ignore_placemarks: list[str]
    input_directory: str
    output_directory: str
    debug_output_directory: str
    output_name_prefix: str
    threshold_meters: float
    debug_enabled: bool
    rename_spans_from_nodes: bool
    merge_contiguous_spans: bool
    merge_contiguous_spans_precision: int
    validate_output: bool = False

    def kml_path(self) -> str:
        """Full path to the KML file."""
        directory = os.path.join(os.getcwd(), self.input_directory)
        return os.path.join(directory, self.kml_file_name)

    def output_paths(self) -> OutputPaths:
        """Resolved output file paths with date suffix."""
        date_string = datetime.today().strftime("%d%b%Y").lower()
        prefix = self.output_name_prefix
        out_dir = self.output_directory

        return OutputPaths(
            nodes_geojson=os.path.join(
                out_dir, f"{prefix}_ofds-nodes_{date_string}.geojson"
            ),
            spans_geojson=os.path.join(
                out_dir, f"{prefix}_ofds-spans_{date_string}.geojson"
            ),
            ofds_json=os.path.join(
                out_dir, f"{prefix}_ofds-json_{date_string}.json"
            ),
            debug_prefix=prefix,
        )


def _case_preserving_config_parser() -> type[configparser.ConfigParser]:
    """Create a ConfigParser that preserves option case."""

    class CasePreservingConfigParser(configparser.ConfigParser):
        def optionxform(self, optionstr: str) -> str:
            return optionstr

    return CasePreservingConfigParser


def load_config(config_file: str) -> Config:
    """Load and validate configuration from an INI file."""

    parser_class = _case_preserving_config_parser()
    config = parser_class()
    config.read(config_file)

    parsed: dict[str, str] = {}
    for option, value in config.defaults().items():
        parsed[option] = value

    for section in config.sections():
        for option in config.options(section):
            parsed[option] = config.get(section, option)

    # Required
    kml_file_name = parsed.get("kml_file_name") or None
    if not kml_file_name:
        print("Error. Please set kml file name in network profile")
        sys.exit(1)

    # Network
    network_name = parsed.get("network_name") or "Default Network Name"
    if network_name == "Default Network Name":
        print("Network name not found in config file. Using default value.")

    network_id = parsed.get("network_id") or str(uuid.uuid4())
    network_link_url = parsed.get("network_links") or DEFAULT_NETWORK_LINKS
    if network_link_url == DEFAULT_NETWORK_LINKS:
        print("Network links not found in config file. Using default value.")

    network_links = [{"rel": "describedby", "href": network_link_url}]

    network_status = parsed.get("network_status") or "Operational"

    # Providers
    physical_infrastructure_provider_id = parsed.get(
        "physicalInfrastructureProvider_id", ""
    )
    physical_infrastructure_provider_name = parsed.get(
        "physicalInfrastructureProvider_name", ""
    )
    network_providers_id = parsed.get("networkProviders_id", "")
    network_providers_name = parsed.get("networkProviders_name", "")

    # Ignore placemarks
    ignore_str = parsed.get("ignore_placemarks", "")
    ignore_placemarks = ignore_str.split(";") if ignore_str else []

    # Directories
    input_directory = parsed.get("input_directory", "input/")
    output_directory = parsed.get("output_directory", "output/")
    debug_output_directory = parsed.get(
        "debug_output_directory", output_directory
    )

    # Output prefix
    network_filename_normalised = kml_file_name.replace(" ", "_").upper()
    output_name_prefix = parsed.get("output_name_prefix") or network_filename_normalised[3:]

    # Threshold
    threshold_str = parsed.get("threshold_meters", "5000")
    try:
        threshold_meters = float(threshold_str)
    except (ValueError, TypeError):
        print(
            f"Warning: Invalid threshold_meters value '{threshold_str}'. "
            "Using default 5000."
        )
        threshold_meters = 5000.0

    # Booleans
    def parse_bool(s: Optional[str], default: bool = False) -> bool:
        if s is None:
            return default
        return str(s).lower() in ("true", "1", "yes", "on")

    debug_enabled = parse_bool(parsed.get("debug_enabled"), False)
    rename_spans_from_nodes = parse_bool(
        parsed.get("rename_spans_from_nodes"), False
    )
    merge_contiguous_spans = parse_bool(
        parsed.get("merge_contiguous_spans"), False
    )
    validate_output = parse_bool(parsed.get("validate_output"), False)

    merge_precision_str = parsed.get(
        "merge_contiguous_spans_precision", "6"
    )
    try:
        merge_contiguous_spans_precision = int(merge_precision_str)
    except (ValueError, TypeError):
        merge_contiguous_spans_precision = 6

    return Config(
        kml_file_name=kml_file_name,
        network_id=network_id,
        network_name=network_name,
        network_status=network_status,
        network_links=network_links,
        physical_infrastructure_provider_id=physical_infrastructure_provider_id,
        physical_infrastructure_provider_name=physical_infrastructure_provider_name,
        network_providers_id=network_providers_id,
        network_providers_name=network_providers_name,
        ignore_placemarks=ignore_placemarks,
        input_directory=input_directory,
        output_directory=output_directory,
        debug_output_directory=debug_output_directory,
        output_name_prefix=output_name_prefix,
        threshold_meters=threshold_meters,
        debug_enabled=debug_enabled,
        rename_spans_from_nodes=rename_spans_from_nodes,
        merge_contiguous_spans=merge_contiguous_spans,
        merge_contiguous_spans_precision=merge_contiguous_spans_precision,
        validate_output=validate_output,
    )
