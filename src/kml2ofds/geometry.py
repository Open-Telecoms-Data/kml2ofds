"""
Geometry and topology operations: snap, break, merge, add nodes, etc.
"""

import json
import uuid
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import (
    Point,
    LineString,
    MultiPoint,
    MultiLineString,
    MultiPolygon,
    GeometryCollection,
)
from shapely.ops import split, nearest_points, unary_union
from shapely.strtree import STRtree

from .constants import ToleranceConfig, AUTO_GENERATED_NODE_NAME, FEATURE_TYPE_NODE


def snap_to_line(
    point: Point,
    lines: gpd.GeoDataFrame,
    tolerance: float = ToleranceConfig.NODE_SNAP,
) -> Optional[Point]:
    """Find the nearest line to a point and return the nearest point on that line."""
    nearest_line = None
    min_distance = float("inf")
    nearest_point_on_line = None

    for line in lines.geometry:
        point_on_line = nearest_points(point, line)[1]
        distance = point.distance(point_on_line)
        if distance < min_distance:
            min_distance = distance
            nearest_line = line
            nearest_point_on_line = point_on_line

    if nearest_line is not None and nearest_point_on_line is not None:
        start_point = nearest_line.coords[0]
        end_point = nearest_line.coords[-1]
        start_buffer = Point(start_point).buffer(tolerance)
        end_buffer = Point(end_point).buffer(tolerance)
        if nearest_point_on_line.within(start_buffer):
            nearest_point_on_line = Point(start_point)
        elif nearest_point_on_line.within(end_buffer):
            nearest_point_on_line = Point(end_point)

    return nearest_point_on_line


def parse_span_endpoint(
    endpoint,
    gdf_nodes: Optional[gpd.GeoDataFrame] = None,
) -> Optional[dict]:
    """Parse span endpoint. Handles node ID string, JSON string, or dict.
    When endpoint is a plain node ID and gdf_nodes is provided, looks up full node info."""
    if endpoint is None:
        return None
    if isinstance(endpoint, dict):
        return endpoint
    if isinstance(endpoint, str):
        try:
            parsed = json.loads(endpoint)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        # Plain node ID string - look up if gdf_nodes provided
        if gdf_nodes is not None and len(gdf_nodes) > 0:
            match = gdf_nodes[gdf_nodes["id"] == endpoint]
            if not match.empty:
                row = match.iloc[0]
                return {
                    "id": endpoint,
                    "name": row.get("name"),
                    "location": {
                        "type": "Point",
                        "coordinates": [row.geometry.x, row.geometry.y],
                    },
                }
        return {"id": endpoint, "name": None, "location": None}
    return None


def find_end_point(
    span_endpoint,
    gdf_nodes: gpd.GeoDataFrame,
    tolerance: float = ToleranceConfig.ENDPOINT_MATCH,
) -> Optional[gpd.GeoSeries]:
    """Find the node matching a span endpoint within tolerance. Uses spatial index."""
    point_geom = Point(span_endpoint)
    buffered_point = point_geom.buffer(tolerance)

    # Use spatial index for faster lookup
    if len(gdf_nodes) == 0:
        return None
    tree = STRtree(gdf_nodes.geometry)
    possible_indices = tree.query(buffered_point)
    if possible_indices is None or len(possible_indices) == 0:
        return None

    matched = gdf_nodes.iloc[possible_indices]
    within_buffer = matched[matched.geometry.within(buffered_point)]
    if within_buffer.empty:
        return None

    distances = within_buffer.geometry.apply(lambda g: point_geom.distance(g))
    closest_idx = distances.idxmin()
    return gdf_nodes.loc[closest_idx]


def append_node(
    new_node_coords,
    network_id: str,
    network_name: str,
    network_links: list,
    physical_infrastructure_provider_id: str,
    physical_infrastructure_provider_name: str,
    network_providers_id: str,
    network_providers_name: str,
) -> dict:
    """Create a GeoJSON feature for an auto-generated node."""
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": new_node_coords},
        "properties": {
            "id": str(uuid.uuid4()),
            "name": AUTO_GENERATED_NODE_NAME,
            "network": {
                "id": network_id,
                "name": network_name,
                "links": network_links,
            },
            "physicalInfrastructureProvider": {
                "id": physical_infrastructure_provider_id,
                "name": physical_infrastructure_provider_name,
            },
            "networkProviders": [
                {"id": network_providers_id, "name": network_providers_name}
            ],
            "featureType": FEATURE_TYPE_NODE,
        },
    }


def convert_to_serializable(obj):
    """Convert dict/list to JSON-serializable form (numpy -> native)."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(x) for x in obj]
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def filter_ignored_nodes(
    gdf_nodes: gpd.GeoDataFrame,
    ignore_placemarks: list[str],
) -> gpd.GeoDataFrame:
    """Filter out nodes matching any ignore pattern."""
    import re

    if not ignore_placemarks:
        return gdf_nodes

    mask = pd.Series([True] * len(gdf_nodes), index=gdf_nodes.index)
    for idx, row in gdf_nodes.iterrows():
        name = row.get("name", "")
        if name:
            for pattern in ignore_placemarks:
                if re.search(rf"{pattern}", name):
                    mask[idx] = False
                    break
    return gdf_nodes[mask].copy()


def extract_node_id(node_json_str) -> Optional[str]:
    """Extract node ID from JSON string, dict, or plain ID string."""
    if node_json_str is None:
        return None
    if isinstance(node_json_str, str):
        try:
            d = json.loads(node_json_str)
            return d.get("id") if isinstance(d, dict) else node_json_str
        except (json.JSONDecodeError, TypeError):
            return node_json_str  # Plain node ID
    if isinstance(node_json_str, dict):
        return node_json_str.get("id")
    return None


# OFDS status values for nodes and spans (nodeStatus/spanStatus codelists)
_OFDS_STATUS_MAP = {
    "operational": "operational",
    "underconstruction": "underConstruction",
    "planned": "planned",
    "decommissioned": "decommissioned",
    "proposed": "proposed",
    "inactive": "inactive",
}


def _network_status_to_ofds(status: str) -> str:
    """Map profile network_status to OFDS schema format."""
    key = (status or "").strip().lower().replace(" ", "")
    return _OFDS_STATUS_MAP.get(key, "operational")


def apply_network_status(
    gdf_nodes: gpd.GeoDataFrame,
    gdf_spans: gpd.GeoDataFrame,
    network_status: str,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Set status on all nodes and spans from network_status config."""
    ofds_status = _network_status_to_ofds(network_status)
    gdf_nodes = gdf_nodes.copy()
    gdf_spans = gdf_spans.copy()
    gdf_nodes["status"] = ofds_status
    gdf_spans["status"] = ofds_status
    return gdf_nodes, gdf_spans


def apply_provider_info(
    gdf_nodes: gpd.GeoDataFrame,
    gdf_spans: gpd.GeoDataFrame,
    physical_infrastructure_provider_id: str,
    physical_infrastructure_provider_name: str,
    network_providers_id: str,
    network_providers_name: str,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Set physicalInfrastructureProvider and networkProviders on all nodes and spans."""
    physical_infrastructure_provider = {
        "id": physical_infrastructure_provider_id or "",
        "name": physical_infrastructure_provider_name or "",
    }
    network_providers = [
        {"id": network_providers_id or "", "name": network_providers_name or ""}
    ]
    gdf_nodes = gdf_nodes.copy()
    gdf_spans = gdf_spans.copy()
    # Broadcast to each row (pandas requires length to match when assigning sequences)
    gdf_nodes["physicalInfrastructureProvider"] = [
        physical_infrastructure_provider
    ] * len(gdf_nodes)
    gdf_spans["physicalInfrastructureProvider"] = [
        physical_infrastructure_provider
    ] * len(gdf_spans)
    gdf_nodes["networkProviders"] = [network_providers] * len(gdf_nodes)
    gdf_spans["networkProviders"] = [network_providers] * len(gdf_spans)
    return gdf_nodes, gdf_spans


def find_self_intersection(line) -> Optional[MultiPoint]:
    """Find points where a LineString self-intersects."""
    if line.is_simple:
        return None
    intersection = unary_union(line)
    seg_coords = []
    if isinstance(intersection, (GeometryCollection, MultiLineString)):
        for seg in intersection.geoms:
            seg_coords.extend(list(seg.coords))
    else:
        seg_coords.extend(list(intersection.coords))
    dup_points = [
        Point(p) for p, c in Counter(seg_coords).items() if c > 1
    ]
    return MultiPoint(dup_points) if dup_points else None


def rejoin_self_intersection_breaks(split_lines, intersect_points) -> GeometryCollection:
    """Rejoin split lines at self-intersection points."""
    joined = []
    i = 0
    geoms = list(split_lines.geoms) if hasattr(split_lines, "geoms") else [split_lines]

    while i < len(geoms):
        current = geoms[i]
        if i + 1 < len(geoms):
            next_line = geoms[i + 1]
            pt_check = Point(next_line.coords[0])
            if (
                current.coords[-1] == next_line.coords[0]
                and intersect_points is not None
                and hasattr(intersect_points, "contains")
                and intersect_points.contains(pt_check)
            ):
                joined_line = LineString(
                    list(current.coords)[:-1] + list(next_line.coords)[1:]
                )
                i += 1
                current = geoms[i]
                while (
                    i + 1 < len(geoms)
                    and current.coords[-1] == geoms[i + 1].coords[0]
                    and intersect_points.contains(Point(geoms[i + 1].coords[0]))
                ):
                    next_line = geoms[i + 1]
                    joined_line = LineString(
                        list(joined_line.coords)[:-1] + list(next_line.coords)[1:]
                    )
                    i += 1
                    current = geoms[i]
                joined.append(joined_line)
            else:
                joined.append(current)
        else:
            joined.append(current)
        i += 1

    return GeometryCollection(joined)


def break_spans_at_node_points(
    gdf_nodes: gpd.GeoDataFrame,
    gdf_spans: gpd.GeoDataFrame,
    network_name: str,
    network_id: str,
    network_links: list,
    buffer_size: float = ToleranceConfig.BUFFER_SIZE,
) -> gpd.GeoDataFrame:
    """Break spans at every node intersection."""
    split_lines = []
    self_intersects = []
    feature_type = "span"

    # Build spatial index for nodes
    node_tree = STRtree(gdf_nodes.geometry) if len(gdf_nodes) > 0 else None

    for _, line_row in gdf_spans.iterrows():
        span_name = line_row["name"]
        line_geom = line_row.geometry

        # Find intersecting nodes using spatial index
        if node_tree is not None:
            possible = node_tree.query(line_geom)
            indices = np.atleast_1d(possible).tolist() if possible is not None else []
            intersected_points = []
            for idx in indices:
                pt = gdf_nodes.geometry.iloc[idx]
                buf_pt = pt.buffer(buffer_size)
                if line_geom.intersects(buf_pt):
                    intersected_points.append(pt)
        else:
            intersected_points = []

        if intersected_points:
            buffered_area = MultiPolygon([p.buffer(buffer_size) for p in intersected_points])

            if line_geom.is_simple:
                split_line = split(line_geom, buffered_area)
            else:
                self_intersect = find_self_intersection(line_geom)
                if self_intersect is not None:
                    self_intersects.append(self_intersect)
                split_line = split(line_geom, buffered_area)
                split_line = rejoin_self_intersection_breaks(
                    split_line, self_intersect
                )

            for segment in split_line.geoms:
                if len(segment.coords) >= 2:
                    split_lines.append(
                        (
                            str(uuid.uuid4()),
                            segment,
                            span_name,
                            feature_type,
                        )
                    )
        else:
            if len(line_geom.coords) >= 2:
                split_lines.append(
                    (
                        str(uuid.uuid4()),
                        line_geom,
                        span_name,
                        feature_type,
                    )
                )

    gdf_spans = gpd.GeoDataFrame(
        split_lines,
        columns=["id", "geometry", "name", "featureType"],
        crs=gdf_spans.crs,
    )

    gdf_spans["network"] = gdf_spans.apply(
        lambda _: {"id": network_id, "name": network_name, "links": network_links},
        axis=1,
    )

    return gdf_spans


def merge_contiguous_spans(
    gdf_spans: gpd.GeoDataFrame,
    precision: int = 6,
) -> gpd.GeoDataFrame:
    """Merge spans that share endpoints (within coordinate precision)."""
    if len(gdf_spans) == 0:
        return gdf_spans

    def round_coord(c):
        return (round(c[0], precision), round(c[1], precision))

    current = gdf_spans.copy()
    for _ in range(100):
        merged_list = []
        processed = set()
        endpoint_map = {}

        for idx, row in current.iterrows():
            sc = round_coord(row.geometry.coords[0])
            ec = round_coord(row.geometry.coords[-1])
            endpoint_map.setdefault(sc, []).append((idx, True))
            endpoint_map.setdefault(ec, []).append((idx, False))

        merges_found = False
        for idx, row in current.iterrows():
            if idx in processed:
                continue
            merged_coords = list(row.geometry.coords)
            merged_span = row.copy()

            extended = True
            while extended:
                extended = False
                sc = round_coord(merged_coords[0])
                ec = round_coord(merged_coords[-1])

                for other_idx, is_other_start in endpoint_map.get(sc, [])[:]:
                    if other_idx == idx or other_idx in processed:
                        continue
                    other = current.loc[other_idx]
                    oec = round_coord(other.geometry.coords[-1])
                    if not is_other_start and oec == sc:
                        coords = list(other.geometry.coords)
                        merged_coords = coords[:-1] + merged_coords
                        processed.add(other_idx)
                        merges_found = True
                        extended = True
                        break

                for other_idx, is_other_start in endpoint_map.get(ec, [])[:]:
                    if other_idx == idx or other_idx in processed:
                        continue
                    other = current.loc[other_idx]
                    osc = round_coord(other.geometry.coords[0])
                    if is_other_start and osc == ec:
                        coords = list(other.geometry.coords)
                        merged_coords = merged_coords + coords[1:]
                        processed.add(other_idx)
                        merges_found = True
                        extended = True
                        break

            if len(merged_coords) >= 2:
                merged_span["geometry"] = LineString(merged_coords)
            merged_list.append(merged_span)
            processed.add(idx)

        if not merges_found:
            break
        current = gpd.GeoDataFrame(merged_list, crs=current.crs)

    return current


def add_missing_nodes(
    gdf_spans: gpd.GeoDataFrame,
    gdf_nodes: gpd.GeoDataFrame,
    network_id: str,
    network_name: str,
    network_links: list,
    physical_infrastructure_provider_id: str,
    physical_infrastructure_provider_name: str,
    network_providers_id: str,
    network_providers_name: str,
    tolerance: float = ToleranceConfig.ENDPOINT_MATCH,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Ensure each span has start and end nodes; add auto-generated nodes if missing."""
    new_nodes = []
    new_nodes_geoms = []

    # Use spatial index for existing nodes
    if len(gdf_nodes) > 0:
        node_tree = STRtree(gdf_nodes.geometry)
    else:
        node_tree = None

    for _, row in gdf_spans.iterrows():
        start_pt = row.geometry.coords[0]
        end_pt = row.geometry.coords[-1]
        start_buf = Point(start_pt).buffer(tolerance)
        end_buf = Point(end_pt).buffer(tolerance)

        def exists_near(pt_buf):
            if node_tree is None:
                return False
            indices = node_tree.query(pt_buf)
            if indices is None or len(indices) == 0:
                return False
            return any(
                gdf_nodes.geometry.iloc[i].intersects(pt_buf)
                for i in indices
            )

        start_exists = exists_near(start_buf)
        end_exists = exists_near(end_buf)

        def add_if_new(coord):
            geom = Point(coord)
            for g in new_nodes_geoms:
                if geom.distance(g) <= tolerance:
                    return
            new_nodes.append(
                append_node(
                    coord,
                    network_id,
                    network_name,
                    network_links,
                    physical_infrastructure_provider_id,
                    physical_infrastructure_provider_name,
                    network_providers_id,
                    network_providers_name,
                )
            )
            new_nodes_geoms.append(geom)

        if not start_exists:
            add_if_new(start_pt)
        if not end_exists:
            add_if_new(end_pt)

    if new_nodes:
        new_gdf = gpd.GeoDataFrame.from_features(new_nodes, crs=gdf_nodes.crs)
        combined = pd.concat([gdf_nodes, new_gdf], ignore_index=True)
    else:
        combined = gdf_nodes
        new_gdf = gpd.GeoDataFrame(
            {c: pd.Series(dtype=gdf_nodes[c].dtype) for c in gdf_nodes.columns}
        ).set_crs(gdf_nodes.crs)

    return combined, new_gdf


def add_nodes_to_spans(
    gdf_spans: gpd.GeoDataFrame,
    gdf_nodes: gpd.GeoDataFrame,
    tolerance: float = ToleranceConfig.ENDPOINT_MATCH,
) -> gpd.GeoDataFrame:
    """Attach start and end node IDs to each span (OFDS expects node IDs, not full objects)."""
    start_ids = []
    end_ids = []

    for counter, (_, span) in enumerate(gdf_spans.iterrows(), 1):
        start_coord = span.geometry.coords[0]
        end_coord = span.geometry.coords[-1]

        start_match = find_end_point(start_coord, gdf_nodes, tolerance)
        end_match = find_end_point(end_coord, gdf_nodes, tolerance)

        start_ids.append(start_match["id"] if start_match is not None else None)
        end_ids.append(end_match["id"] if end_match is not None else None)
        if counter % 100 == 0 or counter == len(gdf_spans):
            print(f"\rAssociating nodes with spans {counter} of {len(gdf_spans)}", end="", flush=True)

    print()
    gdf_spans = gdf_spans.copy()
    gdf_spans["start"] = start_ids
    gdf_spans["end"] = end_ids
    return gdf_spans
