"""
Post-export migration helpers for OFDS package outputs.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass

from .config import OutputPaths

OFDS_03_SCHEMA_URL = (
    "https://raw.githubusercontent.com/Open-Telecoms-Data/"
    "open-fibre-data-standard/0__3__0/schema/network-schema.json"
)
OFDS_04_SCHEMA_URL = (
    "https://standard.ofds.info/en/0__4__0/network-schema.json"
)


@dataclass
class MigrationResult:
    """Counts of transformed fields for reporting/debugging."""

    schema_links_updated: int = 0
    node_provider_fields_migrated: int = 0
    span_provider_fields_migrated: int = 0
    span_deployment_details_migrated: int = 0


def _migrate_links_to_04(network: dict, result: MigrationResult) -> None:
    links = network.get("links")
    if not isinstance(links, list):
        return

    for link in links:
        if not isinstance(link, dict):
            continue
        if (
            link.get("rel") == "describedby"
            and link.get("href") == OFDS_03_SCHEMA_URL
        ):
            link["href"] = OFDS_04_SCHEMA_URL
            result.schema_links_updated += 1


def _migrate_entity_provider_fields(entity: dict) -> bool:
    """
    Move 0.3 `physicalInfrastructureProvider` to 0.4 `transmissionMediumOwner`.

    Returns:
        True if the old field existed and was migrated/removed.
    """
    if "physicalInfrastructureProvider" not in entity:
        return False

    provider = entity.pop("physicalInfrastructureProvider")
    if (
        provider
        and isinstance(provider, dict)
        and "transmissionMediumOwner" not in entity
    ):
        entity["transmissionMediumOwner"] = provider
    return True


def _migrate_span_deployment_details(span: dict) -> bool:
    """
    Move legacy deployment description under 0.4 `supportingInfrastructure`.
    """
    details = span.pop("deploymentDetails", None)
    if not isinstance(details, dict):
        return False

    description = details.get("description")
    if not description:
        return True

    supporting = span.get("supportingInfrastructure")
    if not isinstance(supporting, dict):
        supporting = {}
        span["supportingInfrastructure"] = supporting
    supporting.setdefault("description", description)
    return True


def migrate_package_03_to_04(ofds_json: dict) -> tuple[dict, MigrationResult]:
    """
    Migrate an OFDS package dict from 0.3-era fields to 0.4-compatible fields.
    """
    migrated = deepcopy(ofds_json)
    result = MigrationResult()

    networks = migrated.get("networks")
    if not isinstance(networks, list):
        return migrated, result

    for network in networks:
        if not isinstance(network, dict):
            continue
        _migrate_links_to_04(network, result)

        nodes = network.get("nodes", [])
        if isinstance(nodes, list):
            for node in nodes:
                if (
                    isinstance(node, dict)
                    and _migrate_entity_provider_fields(node)
                ):
                    result.node_provider_fields_migrated += 1

        spans = network.get("spans", [])
        if isinstance(spans, list):
            for span in spans:
                if not isinstance(span, dict):
                    continue
                if _migrate_entity_provider_fields(span):
                    result.span_provider_fields_migrated += 1
                if _migrate_span_deployment_details(span):
                    result.span_deployment_details_migrated += 1

    return migrated, result


def migrate_output_files_to_04(
    paths: OutputPaths,
    validate: bool,
) -> MigrationResult:
    """
    Upgrade generated output files to OFDS 0.4.

    Reads OFDS JSON file, migrates fields, validates against current libcove
    schema when requested, then regenerates GeoJSON from migrated package.
    """
    with open(paths.ofds_json, encoding="utf-8") as f:
        ofds_json = json.load(f)

    migrated, result = migrate_package_03_to_04(ofds_json)

    if validate:
        from libcoveofds.python_validate import PythonValidate
        from libcoveofds.schema import OFDSSchema

        schema = OFDSSchema()
        validator = PythonValidate(schema)
        validation_errors = validator.validate(migrated)
        if validation_errors:
            shown = "\n".join(str(err) for err in validation_errors[:10])
            remainder = max(len(validation_errors) - 10, 0)
            if remainder:
                shown = f"{shown}\n... and {remainder} more errors"
            raise ValueError(
                f"OFDS 0.4 validation failed after migration:\n{shown}"
            )

    from libcoveofds.geojson import JSONToGeoJSONConverter

    with open(paths.ofds_json, "w", encoding="utf-8") as f:
        json.dump(migrated, f, indent=4, ensure_ascii=False)

    # JSONToGeoJSONConverter mutates package in place (popping nodes/spans),
    # so convert from a copy after writing the canonical JSON output.
    converter = JSONToGeoJSONConverter()
    converter.process_package(deepcopy(migrated))
    with open(paths.nodes_geojson, "w", encoding="utf-8") as f:
        json.dump(
            converter.get_nodes_geojson(),
            f,
            indent=4,
            ensure_ascii=False,
        )
    with open(paths.spans_geojson, "w", encoding="utf-8") as f:
        json.dump(
            converter.get_spans_geojson(),
            f,
            indent=4,
            ensure_ascii=False,
        )

    return result
