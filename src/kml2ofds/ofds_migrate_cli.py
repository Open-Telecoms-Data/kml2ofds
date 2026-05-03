"""
CLI utility to convert OFDS 0.3 JSON datasets to OFDS 0.4.
"""

from __future__ import annotations

import json

import click

from .ofds_migrate import migrate_package_03_to_04


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)


@click.command(help="Convert an OFDS JSON dataset from 0.3 to 0.4.")
@click.option(
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to input OFDS 0.3 JSON file.",
)
@click.option(
    "--output",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False),
    help="Path to output OFDS 0.4 JSON file.",
)
@click.option(
    "--nodes-geojson-output",
    "nodes_geojson_output",
    type=click.Path(dir_okay=False),
    default=None,
    help="Optional path to write regenerated nodes GeoJSON.",
)
@click.option(
    "--spans-geojson-output",
    "spans_geojson_output",
    type=click.Path(dir_okay=False),
    default=None,
    help="Optional path to write regenerated spans GeoJSON.",
)
@click.option(
    "--validate/--no-validate",
    default=True,
    show_default=True,
    help="Validate migrated JSON with libcoveofds.",
)
def main(
    input_path: str,
    output_path: str,
    nodes_geojson_output: str | None,
    spans_geojson_output: str | None,
    validate: bool,
) -> None:
    """
    Convert an OFDS package file from 0.3-era fields to 0.4-compatible fields.
    """
    with open(input_path, encoding="utf-8") as f:
        source = json.load(f)

    migrated, result = migrate_package_03_to_04(source)

    if validate:
        from libcoveofds.python_validate import PythonValidate
        from libcoveofds.schema import OFDSSchema

        schema = OFDSSchema()
        validator = PythonValidate(schema)
        errors = validator.validate(migrated)
        if errors:
            shown = "\n".join(str(err) for err in errors[:10])
            remainder = max(len(errors) - 10, 0)
            if remainder:
                shown = f"{shown}\n... and {remainder} more errors"
            raise click.ClickException(
                f"Migrated output failed OFDS validation:\n{shown}"
            )

    _write_json(output_path, migrated)

    if nodes_geojson_output or spans_geojson_output:
        from libcoveofds.geojson import JSONToGeoJSONConverter

        from .libcoveofds_span_endpoint_patch import apply_libcoveofds_span_endpoint_patch

        apply_libcoveofds_span_endpoint_patch()

        converter = JSONToGeoJSONConverter()
        converter.process_package(migrated)
        if nodes_geojson_output:
            _write_json(nodes_geojson_output, converter.get_nodes_geojson())
        if spans_geojson_output:
            _write_json(spans_geojson_output, converter.get_spans_geojson())

    click.echo("Converted OFDS dataset to 0.4-compatible fields.")
    click.echo(
        "Changes: "
        f"links={result.schema_links_updated}, "
        f"nodes={result.node_provider_fields_migrated}, "
        f"spans={result.span_provider_fields_migrated}, "
        f"deployment_details={result.span_deployment_details_migrated}"
    )
