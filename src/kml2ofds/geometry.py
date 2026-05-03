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
from scipy.spatial import cKDTree
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


def meters_to_planar_degrees(meters: float) -> float:
    """Approximate metres to degrees (latitude scale, same as consolidation)."""
    return float(meters) / 111000.0


def merge_proximate_nodes(
    gdf_nodes: gpd.GeoDataFrame,
    threshold_meters: float,
) -> gpd.GeoDataFrame:
    """Merge nodes within ``threshold_meters`` (planar distance in WGS84 degrees).

    Uses the same degree-per-metre approximation as elsewhere in the package.
    Combined ``name`` is unique original names in row order, joined with ``" / "``.
    Merged geometry is the centroid. Each merged cluster gets a new ``id``; singletons
    keep their properties.
    """
    if len(gdf_nodes) <= 1:
        return gdf_nodes.copy()

    gdf = gdf_nodes.reset_index(drop=True)
    n = len(gdf)
    coords = np.array(
        [(gdf.geometry.iloc[i].x, gdf.geometry.iloc[i].y) for i in range(n)],
        dtype=float,
    )
    r_deg = meters_to_planar_degrees(threshold_meters)
    if r_deg <= 0:
        return gdf_nodes.copy()

    tree = cKDTree(coords)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in tree.query_ball_point(coords[i], r=r_deg):
            j = int(j)
            if j > i:
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        r = find(i)
        clusters.setdefault(r, []).append(i)

    chunks: list[gpd.GeoDataFrame] = []
    for _root, members in sorted(clusters.items(), key=lambda kv: min(kv[1])):
        members.sort()
        if len(members) == 1:
            chunks.append(gdf.iloc[[members[0]]])
            continue
        sub = gdf.iloc[members]
        cx = float(sub.geometry.x.mean())
        cy = float(sub.geometry.y.mean())
        names_ordered: list[str] = []
        seen: set[str] = set()
        for k in members:
            raw = gdf.iloc[k].get("name")
            s = "" if raw is None else str(raw).strip()
            if s and s not in seen:
                seen.add(s)
                names_ordered.append(s)
        merged_name = " / ".join(names_ordered)
        row0 = gdf.iloc[members[0]].copy()
        row0["geometry"] = Point(cx, cy)
        row0["name"] = merged_name
        row0["id"] = str(uuid.uuid4())
        chunks.append(gpd.GeoDataFrame([row0.to_dict()], crs=gdf.crs))

    out = pd.concat(chunks, ignore_index=True)
    return gpd.GeoDataFrame(out, geometry="geometry", crs=gdf_nodes.crs)


def snap_nodes_to_spans(
    gdf_nodes: gpd.GeoDataFrame,
    gdf_spans: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Snap every node point to the nearest span geometry (see ``snap_to_line``)."""
    if len(gdf_nodes) == 0:
        return gdf_nodes.copy()
    snapped = gdf_nodes.geometry.map(lambda p: snap_to_line(p, gdf_spans))
    out = gpd.GeoDataFrame(gdf_nodes.drop(columns="geometry").copy())
    out["geometry"] = snapped
    out.set_geometry("geometry", inplace=True)
    return out


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
    """Break spans where each node lies within buffer_size of the line (degrees in CRS).

    Default is ``ToleranceConfig.NODE_SNAP`` so breaks align with snap-to-line tolerance.
    """
    split_lines = []
    self_intersects = []
    feature_type = "span"

    # Build spatial index for nodes
    node_tree = STRtree(gdf_nodes.geometry) if len(gdf_nodes) > 0 else None

    for _, line_row in gdf_spans.iterrows():
        span_name = line_row["name"]
        line_geom = line_row.geometry

        # STRtree.query(line) uses bbox overlap only. Axis-aligned spans have a
        # degenerate envelope, so a point with tiny float offset from the line
        # (e.g. y=1e-12 on a horizontal segment) is missed. Query buffered line
        # with predicate intersects instead.
        if node_tree is not None:
            query_geom = line_geom.buffer(buffer_size)
            possible = node_tree.query(
                query_geom, predicate="intersects"
            )
            indices = (
                np.atleast_1d(possible).tolist()
                if possible is not None
                else []
            )
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
    endpoint_tolerance: float | None = None,
    gdf_nodes: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
    """Merge spans whose endpoints meet within a distance tolerance.

    Matches endpoints using planar distance in the data CRS (typically degrees
    for WGS84), not bitwise equality of rounded coordinates. Supports both
    head-to-tail chains and joins where a common vertex appears as two span
    starts or two span ends (one segment is reversed when splicing).

    In ``run_pipeline`` this runs before ``break_spans_at_node_points``. If it
    ran after a break, it would rejoin split pieces (sharp bends, T-junctions,
    etc.) and undo node-based splits.

    If ``gdf_nodes`` is set, two spans are not merged when a node lies within
    ``tol`` of the junction. That keeps separate KML paths that meet at a marked
    node (e.g. two arms of a tight corner) as separate spans.

    If ``endpoint_tolerance`` is None, it defaults to ``10 ** (-precision)`` so
    existing ``merge_contiguous_spans_precision`` settings keep a similar scale
    (e.g. precision 3 → 1e-3).
    """
    if len(gdf_spans) == 0:
        return gdf_spans

    tol = float(endpoint_tolerance if endpoint_tolerance is not None else 10.0 ** (-precision))

    node_tree = None
    if gdf_nodes is not None and len(gdf_nodes) > 0:
        node_xy = np.array(
            [(p.x, p.y) for p in gdf_nodes.geometry],
            dtype=float,
        )
        if len(node_xy) > 0:
            node_tree = cKDTree(node_xy)

    def _junction_has_node(junction: tuple) -> bool:
        if node_tree is None:
            return False
        p = np.asarray(junction[:2], dtype=float)
        return len(node_tree.query_ball_point(p, r=tol)) > 0

    def _endpoint_dist(a, b) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    current = gdf_spans.copy()
    for _ in range(100):
        merged_list = []
        processed = set()
        items = list(current.iterrows())
        endpoints: list[tuple[object, bool, np.ndarray]] = []
        for idx, row in items:
            c = np.asarray(row.geometry.coords, dtype=float)
            endpoints.append((idx, True, c[0].copy()))
            endpoints.append((idx, False, c[-1].copy()))
        pts = np.stack([e[2] for e in endpoints], axis=0)
        tree = cKDTree(pts)

        def _pick_prepend(
            merged_coords: list, idx: object, processed: set
        ) -> tuple[list | None, object | None]:
            p = merged_coords[0]
            p_arr = np.asarray(p, dtype=float)
            best: tuple[float, object, str] | None = None
            for ei in tree.query_ball_point(p_arr, r=tol):
                oidx, is_start, pos = endpoints[ei]
                if oidx == idx or oidx in processed:
                    continue
                d = _endpoint_dist(pos, p)
                if d > tol:
                    continue
                if _junction_has_node(tuple(p)):
                    continue
                kind = "prepend_rev" if is_start else "prepend_fwd"
                if best is None or d < best[0]:
                    best = (d, oidx, kind)
            if best is None:
                return None, None
            _, oidx, kind = best
            oc = list(current.loc[oidx].geometry.coords)
            if kind == "prepend_fwd":
                j = tuple(oc[-1])
                new_coords = [tuple(t) for t in oc[:-1]] + [j] + [
                    tuple(merged_coords[i]) for i in range(1, len(merged_coords))
                ]
            else:
                rev = list(reversed(oc))
                j = tuple(rev[-1])
                new_coords = [tuple(t) for t in rev[:-1]] + [j] + [
                    tuple(merged_coords[i]) for i in range(1, len(merged_coords))
                ]
            return new_coords, oidx

        def _pick_append(
            merged_coords: list, idx: object, processed: set
        ) -> tuple[list | None, object | None]:
            p = merged_coords[-1]
            p_arr = np.asarray(p, dtype=float)
            best: tuple[float, object, str] | None = None
            for ei in tree.query_ball_point(p_arr, r=tol):
                oidx, is_start, pos = endpoints[ei]
                if oidx == idx or oidx in processed:
                    continue
                d = _endpoint_dist(pos, p)
                if d > tol:
                    continue
                if _junction_has_node(tuple(p)):
                    continue
                kind = "append_fwd" if is_start else "append_rev"
                if best is None or d < best[0]:
                    best = (d, oidx, kind)
            if best is None:
                return None, None
            _, oidx, kind = best
            oc = list(current.loc[oidx].geometry.coords)
            j = tuple(merged_coords[-1])
            if kind == "append_fwd":
                new_coords = [tuple(t) for t in merged_coords[:-1]] + [j] + [
                    tuple(oc[i]) for i in range(1, len(oc))
                ]
            else:
                rev = list(reversed(oc))
                new_coords = [tuple(t) for t in merged_coords[:-1]] + [j] + [
                    tuple(rev[i]) for i in range(1, len(rev))
                ]
            return new_coords, oidx

        merges_found = False
        for idx, row in items:
            if idx in processed:
                continue
            merged_coords = [tuple(t) for t in row.geometry.coords]
            merged_span = row.copy()

            extended = True
            while extended:
                extended = False
                new_c, oidx = _pick_prepend(merged_coords, idx, processed)
                if oidx is not None:
                    merged_coords = new_c
                    processed.add(oidx)
                    merges_found = True
                    extended = True
                    continue
                new_c, oidx = _pick_append(merged_coords, idx, processed)
                if oidx is not None:
                    merged_coords = new_c
                    processed.add(oidx)
                    merges_found = True
                    extended = True

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

    crs = gdf_nodes.crs or "EPSG:4326"
    nodes_base = (
        gdf_nodes.set_crs(crs, allow_override=True)
        if gdf_nodes.crs is None
        else gdf_nodes
    )
    if new_nodes:
        new_gdf = gpd.GeoDataFrame.from_features(new_nodes, crs=crs)
        combined = pd.concat([nodes_base, new_gdf], ignore_index=True)
    else:
        combined = nodes_base
        new_gdf = gpd.GeoDataFrame(
            {c: pd.Series(dtype=gdf_nodes[c].dtype) for c in gdf_nodes.columns}
        ).set_crs(crs)

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
            # Newlines (not \\r) so systemd/journald records clean lines.
            print(
                f"  Associating nodes with spans {counter}/{len(gdf_spans)}",
                flush=True,
            )
    gdf_spans = gdf_spans.copy()
    gdf_spans["start"] = start_ids
    gdf_spans["end"] = end_ids
    return gdf_spans
