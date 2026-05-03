"""
Patch libcoveofds JSONToGeoJSONConverter span endpoint resolution.

The upstream implementation resolves each span's start/end by scanning the full
node list (O(spans * nodes)). Large fibre datasets (e.g. national KML exports)
make the OFDS 0.4 migration step appear hung during web "Exporting...".

This module replaces ``_convert_span_to_feature`` with a version that builds a
per-node-list id index once and uses O(1) lookups. Behaviour matches the
original for typical string node ids.
"""

from __future__ import annotations

import copy
from typing import Any

_PATCH_APPLIED = False


def apply_libcoveofds_span_endpoint_patch() -> None:
    """Idempotently monkeypatch JSONToGeoJSONConverter._convert_span_to_feature."""
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    from libcoveofds.geojson import JSONToGeoJSONConverter

    def _convert_span_to_feature(
        self: Any,
        span_data: dict,
        reduced_network_data: dict,
        organisations: list,
        phases: list,
        nodes: list,
    ) -> dict:
        reduced_span_data = copy.deepcopy(span_data)

        feature = {
            "type": "Feature",
            "geometry": reduced_span_data.pop("route")
            if isinstance(reduced_span_data.get("route"), dict)
            else None,
        }

        if isinstance(reduced_span_data.get("physicalInfrastructureProvider"), dict):
            reduced_span_data[
                "physicalInfrastructureProvider"
            ] = self._dereference_object(
                reduced_span_data["physicalInfrastructureProvider"], organisations
            )
        if isinstance(reduced_span_data.get("networkProviders"), list):
            reduced_span_data["networkProviders"] = [
                self._dereference_object(i, organisations)
                for i in reduced_span_data["networkProviders"]
                if isinstance(i, dict)
            ]

        if "phase" in reduced_span_data:
            reduced_span_data["phase"] = self._dereference_object(
                reduced_span_data["phase"], phases
            )

        nodes_key = id(nodes)
        if getattr(self, "_kml2ofds_nodes_by_id_key", None) != nodes_key:
            self._kml2ofds_nodes_by_id = {
                n["id"]: n
                for n in nodes
                if isinstance(n, dict) and "id" in n
            }
            self._kml2ofds_nodes_by_id_key = nodes_key
        nodes_by_id = self._kml2ofds_nodes_by_id

        for endpoint in ["start", "end"]:
            if endpoint not in reduced_span_data:
                continue
            ep_val = reduced_span_data[endpoint]
            try:
                hit = nodes_by_id.get(ep_val)
            except TypeError:
                hit = None
            if hit is not None:
                reduced_span_data[endpoint] = hit
                continue
            for node in nodes:
                if "id" in node and node["id"] == ep_val:
                    reduced_span_data[endpoint] = node
                    break

        feature["properties"] = reduced_span_data
        feature["properties"]["network"] = reduced_network_data
        feature["properties"]["featureType"] = "span"

        return feature

    JSONToGeoJSONConverter._convert_span_to_feature = _convert_span_to_feature
    _PATCH_APPLIED = True
