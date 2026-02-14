"""
Consolidation of auto-generated nodes: merge, split, dedupe, validate.
"""

import json
import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
from scipy.spatial import cKDTree

from .constants import ToleranceConfig, AUTO_GENERATED_NODE_NAME, NETWORK_FORK_NAME
from .geometry import parse_span_endpoint


def _extract_id(x):
    """Extract node ID from span endpoint (dict, JSON string, or other)."""
    d = parse_span_endpoint(x)
    return d.get("id") if d else None


def _span_name_from_endpoints(start_dict, end_dict):
    """Build span name from start and end endpoint names."""
    start_name = start_dict.get("name") if start_dict else None
    end_name = end_dict.get("name") if end_dict else None
    start = start_name or "Unknown"
    end = end_name or "Unknown"
    return f"{start} - {end}"


def write_debug_geojson(
    gdf_nodes: gpd.GeoDataFrame,
    gdf_spans: gpd.GeoDataFrame,
    output_dir: str,
    phase_name: str,
    output_prefix: str,
) -> None:
    """Write debug GeoJSON files for a phase."""
    nodes_file = Path(output_dir) / f"{output_prefix}_debug_{phase_name}_nodes.geojson"
    spans_file = Path(output_dir) / f"{output_prefix}_debug_{phase_name}_spans.geojson"
    gdf_nodes.to_file(nodes_file, driver="GeoJSON")
    gdf_spans.to_file(spans_file, driver="GeoJSON")
    print(f"  [DEBUG] Wrote debug files: {nodes_file.name}, {spans_file.name}")


def _setup_consolidation(gdf_ofds_nodes, gdf_ofds_spans, threshold_meters):
    """Phase 1: Setup and initial filtering."""
    METERS_TO_DEGREES = 1.0 / 111000.0
    threshold = threshold_meters * METERS_TO_DEGREES

    auto_gen_nodes = gdf_ofds_nodes[
        gdf_ofds_nodes["name"] == AUTO_GENERATED_NODE_NAME
    ]
    if len(auto_gen_nodes) == 0:
        print("No auto-generated nodes to process.")
        return None

    other_nodes = gdf_ofds_nodes[
        gdf_ofds_nodes["name"] != AUTO_GENERATED_NODE_NAME
    ]
    start_ids = gdf_ofds_spans["start"].apply(_extract_id)
    end_ids = gdf_ofds_spans["end"].apply(_extract_id)
    span_endpoint_ids = set(pd.concat([start_ids, end_ids]).dropna())

    return threshold, auto_gen_nodes, other_nodes, span_endpoint_ids


def _analyze_auto_generated_nodes(
    auto_gen_nodes, other_nodes, gdf_ofds_spans, span_endpoint_ids
):
    """Phase 2: Analyze auto-generated nodes."""
    print(f"\nAnalyzing {len(auto_gen_nodes)} auto-generated nodes:")
    print("-" * 80)
    for node_idx, node_row in auto_gen_nodes.iterrows():
        node_point = node_row.geometry
        node_id = node_row["id"]
        is_endpoint = node_id in span_endpoint_ids
        min_node_distance = float("inf")
        nearest_node_id = None
        if len(other_nodes) > 0:
            for other_idx, other_row in other_nodes.iterrows():
                distance = node_point.distance(other_row.geometry)
                if distance < min_node_distance:
                    min_node_distance = distance
                    nearest_node_id = other_row["id"]
        min_span_distance = float("inf")
        for span_idx, span_row in gdf_ofds_spans.iterrows():
            nearest_point = nearest_points(node_point, span_row.geometry)[1]
            distance = node_point.distance(nearest_point)
            if distance < min_span_distance:
                min_span_distance = distance
        status = "endpoint" if is_endpoint else "isolated"


def _merge_auto_generated_with_proper_nodes(
    gdf_ofds_nodes, gdf_ofds_spans, threshold
):
    """Phase 3: Merge auto-generated nodes close to proper nodes."""
    coords = np.array([(p.x, p.y) for p in gdf_ofds_nodes.geometry])
    tree = cKDTree(coords)
    clusters = [
        list(idxs) for idxs in tree.query_ball_point(coords, r=threshold)
        if len(idxs) > 1
    ]

    auto_to_proper = {}
    for cluster in clusters:
        auto_indices = []
        proper_indices = []
        for idx in cluster:
            if gdf_ofds_nodes.iloc[idx]["name"] == AUTO_GENERATED_NODE_NAME:
                auto_indices.append(idx)
            else:
                proper_indices.append(idx)
        if auto_indices and proper_indices:
            proper_id = gdf_ofds_nodes.iloc[proper_indices[0]]["id"]
            for auto_idx in auto_indices:
                auto_to_proper[gdf_ofds_nodes.iloc[auto_idx]["id"]] = proper_id

    merged_ids = []
    for index, span in gdf_ofds_spans.iterrows():
        start_dict = parse_span_endpoint(span["start"], gdf_ofds_nodes)
        end_dict = parse_span_endpoint(span["end"], gdf_ofds_nodes)
        start_updated = False
        end_updated = False

        if start_dict and (start_id := start_dict.get("id")) in auto_to_proper:
            proper_id = auto_to_proper[start_id]
            proper_row = gdf_ofds_nodes[gdf_ofds_nodes["id"] == proper_id]
            if not proper_row.empty:
                p = proper_row.iloc[0]
                start_dict["id"] = p["id"]
                start_dict["name"] = p["name"]
                if "location" in start_dict:
                    start_dict["location"]["coordinates"] = [p.geometry.x, p.geometry.y]
                merged_ids.append(start_id)
                start_updated = True
                geom = span["geometry"]
                coords = list(geom.coords)
                coords[0] = (p.geometry.x, p.geometry.y)
                gdf_ofds_spans.at[index, "geometry"] = LineString(coords)

        if end_dict and (end_id := end_dict.get("id")) in auto_to_proper:
            proper_id = auto_to_proper[end_id]
            proper_row = gdf_ofds_nodes[gdf_ofds_nodes["id"] == proper_id]
            if not proper_row.empty:
                p = proper_row.iloc[0]
                end_dict["id"] = p["id"]
                end_dict["name"] = p["name"]
                if "location" in end_dict:
                    end_dict["location"]["coordinates"] = [p.geometry.x, p.geometry.y]
                merged_ids.append(end_id)
                end_updated = True
                # Use current geometry (may have been updated by start block above)
                geom = gdf_ofds_spans.at[index, "geometry"]
                coords = list(geom.coords)
                coords[-1] = (p.geometry.x, p.geometry.y)
                gdf_ofds_spans.at[index, "geometry"] = LineString(coords)

        if start_updated or end_updated:
            gdf_ofds_spans.at[index, "start"] = (
                start_dict["id"] if start_dict else None
            )
            gdf_ofds_spans.at[index, "end"] = (
                end_dict["id"] if end_dict else None
            )
            gdf_ofds_spans.at[index, "name"] = _span_name_from_endpoints(
                start_dict, end_dict
            )

    gdf_ofds_nodes = gdf_ofds_nodes[~gdf_ofds_nodes["id"].isin(merged_ids)]
    print(
        f"Phase 3: Merged {len(set(merged_ids))} auto-generated nodes "
        f"with proper nodes. Remaining nodes: {len(gdf_ofds_nodes)}"
    )
    return gdf_ofds_nodes, gdf_ofds_spans


def _merge_large_auto_generated_clusters(
    gdf_ofds_nodes, gdf_ofds_spans, threshold
):
    """Phase 4: Merge clusters of 3+ auto-generated nodes into fork nodes."""
    auto_gen = gdf_ofds_nodes[
        gdf_ofds_nodes["name"] == AUTO_GENERATED_NODE_NAME
    ]
    coords = np.array([(p.x, p.y) for p in auto_gen.geometry])
    nodes_to_remove = set()
    new_fork_nodes = []

    if len(coords) > 0:
        tree = cKDTree(coords)
        all_clusters = [
            list(c) for c in tree.query_ball_point(coords, r=threshold)
            if len(c) >= 2
        ]
        larger = [c for c in all_clusters if len(c) >= 3]
        unique = list(set(tuple(sorted(c)) for c in larger))

        for cluster in unique:
            cluster_ids = [auto_gen.iloc[i]["id"] for i in cluster]
            pts = [auto_gen.iloc[i].geometry for i in cluster]
            cx = sum(p.x for p in pts) / len(pts)
            cy = sum(p.y for p in pts) / len(pts)
            fork_loc = Point(cx, cy)

            first = auto_gen.iloc[cluster[0]]
            fork_id = str(uuid.uuid4())
            fork_node = {
                "id": fork_id,
                "name": NETWORK_FORK_NAME,
                "geometry": fork_loc,
                "network": first.get("network", {}),
                "physicalInfrastructureProvider": first.get(
                    "physicalInfrastructureProvider", {}
                ),
                "networkProviders": first.get("networkProviders", []),
                "featureType": "node",
            }
            new_fork_nodes.append(fork_node)

            connected = []
            for span_idx, span_row in gdf_ofds_spans.iterrows():
                start_dict = parse_span_endpoint(span_row["start"], gdf_ofds_nodes)
                end_dict = parse_span_endpoint(span_row["end"], gdf_ofds_nodes)
                start_id = start_dict.get("id") if start_dict else None
                end_id = end_dict.get("id") if end_dict else None
                start_name = start_dict.get("name") if start_dict else None
                end_name = end_dict.get("name") if end_dict else None

                if (
                    (start_name and start_name != AUTO_GENERATED_NODE_NAME and start_name != NETWORK_FORK_NAME)
                    or (end_name and end_name != AUTO_GENERATED_NODE_NAME and end_name != NETWORK_FORK_NAME)
                ):
                    continue
                if start_id in cluster_ids or end_id in cluster_ids:
                    connected.append((
                        span_idx, span_row,
                        start_id in cluster_ids,
                        end_id in cluster_ids,
                    ))

            for span_idx, span_row, start_in, end_in in connected:
                start_dict = parse_span_endpoint(span_row["start"], gdf_ofds_nodes)
                end_dict = parse_span_endpoint(span_row["end"], gdf_ofds_nodes)
                geom = span_row["geometry"]
                coords = list(geom.coords)
                updated = False

                if start_in and start_dict:
                    start_dict["id"] = fork_id
                    start_dict["name"] = NETWORK_FORK_NAME
                    if "location" in start_dict:
                        start_dict["location"]["coordinates"] = [cx, cy]
                    coords[0] = (cx, cy)
                    updated = True
                if end_in and end_dict:
                    end_dict["id"] = fork_id
                    end_dict["name"] = NETWORK_FORK_NAME
                    if "location" in end_dict:
                        end_dict["location"]["coordinates"] = [cx, cy]
                    coords[-1] = (cx, cy)
                    updated = True

                if updated:
                    cleaned = [coords[0]]
                    for c in coords[1:]:
                        if c != cleaned[-1]:
                            cleaned.append(c)
                    if len(cleaned) >= 2:
                        gdf_ofds_spans.at[span_idx, "geometry"] = LineString(cleaned)
                        gdf_ofds_spans.at[span_idx, "start"] = (
                            start_dict["id"] if start_dict else None
                        )
                        gdf_ofds_spans.at[span_idx, "end"] = (
                            end_dict["id"] if end_dict else None
                        )
                        gdf_ofds_spans.at[span_idx, "name"] = _span_name_from_endpoints(
                            start_dict, end_dict
                        )

            for nid in cluster_ids:
                nodes_to_remove.add(nid)

        if new_fork_nodes:
            fork_gdf = gpd.GeoDataFrame(new_fork_nodes, crs=gdf_ofds_nodes.crs)
            gdf_ofds_nodes = pd.concat([gdf_ofds_nodes, fork_gdf], ignore_index=True)
        if nodes_to_remove:
            gdf_ofds_nodes = gdf_ofds_nodes[
                ~gdf_ofds_nodes["id"].isin(nodes_to_remove)
            ]
            print(
                f"Phase 4: Processed {len(unique)} clusters (3+ nodes). "
                f"Removed {len(nodes_to_remove)} auto-generated nodes, "
                f"created {len(new_fork_nodes)} network fork nodes. "
                f"Remaining nodes: {len(gdf_ofds_nodes)}"
            )

    return gdf_ofds_nodes, gdf_ofds_spans


def _merge_pair_auto_generated_nodes(
    gdf_ofds_nodes, gdf_ofds_spans, threshold
):
    """Phase 5: Merge pairs of auto-generated nodes by joining spans."""
    auto_gen = gdf_ofds_nodes[
        gdf_ofds_nodes["name"] == AUTO_GENERATED_NODE_NAME
    ]
    coords = np.array([(p.x, p.y) for p in auto_gen.geometry])
    nodes_to_remove = set()
    spans_to_remove = []
    new_spans = []

    if len(coords) > 0:
        tree = cKDTree(coords)
        all_clusters = [
            list(c) for c in tree.query_ball_point(coords, r=threshold)
            if len(c) >= 2
        ]
        pairs = [c for c in all_clusters if len(c) == 2]
        unique_pairs = list(
            set(
                (min(c[0], c[1]), max(c[0], c[1]))
                for c in pairs
            )
        )

        for (ia, ib) in unique_pairs:
            node_a_id = auto_gen.iloc[ia]["id"]
            node_b_id = auto_gen.iloc[ib]["id"]

            spans_a = []
            spans_b = []
            for span_idx, span_row in gdf_ofds_spans.iterrows():
                start_dict = parse_span_endpoint(span_row["start"], gdf_ofds_nodes)
                end_dict = parse_span_endpoint(span_row["end"], gdf_ofds_nodes)
                start_id = start_dict.get("id") if start_dict else None
                end_id = end_dict.get("id") if end_dict else None

                if start_id == node_a_id or end_id == node_a_id:
                    spans_a.append((span_idx, span_row, start_id == node_a_id))
                if start_id == node_b_id or end_id == node_b_id:
                    spans_b.append((span_idx, span_row, start_id == node_b_id))

            unique_spans = {}
            for span_idx, span_row, is_start in spans_a:
                if span_idx not in unique_spans:
                    unique_spans[span_idx] = (span_row, is_start, False)
            for span_idx, span_row, is_start in spans_b:
                if span_idx not in unique_spans:
                    unique_spans[span_idx] = (span_row, False, is_start)
                else:
                    unique_spans[span_idx] = (span_row, True, True)

            if len(unique_spans) == 2:
                items = list(unique_spans.items())
                (s1_idx, (s1_row, c1a, c1b)), (s2_idx, (s2_row, c2a, c2b)) = items

                s1_geom = s1_row["geometry"]
                s2_geom = s2_row["geometry"]
                s1_start = parse_span_endpoint(s1_row["start"], gdf_ofds_nodes)
                s1_end = parse_span_endpoint(s1_row["end"], gdf_ofds_nodes)
                s2_start = parse_span_endpoint(s2_row["start"], gdf_ofds_nodes)
                s2_end = parse_span_endpoint(s2_row["end"], gdf_ofds_nodes)

                s1_conn = None
                s1_other = None
                if s1_start and s1_start.get("id") in (node_a_id, node_b_id):
                    s1_conn = "start"
                    s1_other = s1_end
                elif s1_end and s1_end.get("id") in (node_a_id, node_b_id):
                    s1_conn = "end"
                    s1_other = s1_start

                s2_conn = None
                s2_other = None
                if s2_start and s2_start.get("id") in (node_a_id, node_b_id):
                    s2_conn = "start"
                    s2_other = s2_end
                elif s2_end and s2_end.get("id") in (node_a_id, node_b_id):
                    s2_conn = "end"
                    s2_other = s2_start

                if s1_conn == "start" and s2_conn == "start":
                    geom1 = LineString(list(s1_geom.coords)[::-1])
                    geom2 = s2_geom
                elif s1_conn == "start" and s2_conn == "end":
                    geom1 = LineString(list(s1_geom.coords)[::-1])
                    geom2 = LineString(list(s2_geom.coords)[::-1])
                elif s1_conn == "end" and s2_conn == "start":
                    geom1 = s1_geom
                    geom2 = s2_geom
                elif s1_conn == "end" and s2_conn == "end":
                    geom1 = s1_geom
                    geom2 = LineString(list(s2_geom.coords)[::-1])
                else:
                    geom1 = s1_geom
                    geom2 = LineString(list(s2_geom.coords)[::-1])
                    s1_other = s1_start
                    s2_other = s2_start

                joined = list(geom1.coords) + list(geom2.coords)
                cleaned = [joined[0]]
                for c in joined[1:]:
                    if c != cleaned[-1]:
                        cleaned.append(c)

                if len(cleaned) >= 2:
                    start_id = s1_other.get("id") if s1_other else None
                    end_id = s2_other.get("id") if s2_other else None
                    start_ok = (
                        not start_id
                        or not gdf_ofds_nodes[gdf_ofds_nodes["id"] == start_id].empty
                    )
                    end_ok = (
                        not end_id
                        or not gdf_ofds_nodes[gdf_ofds_nodes["id"] == end_id].empty
                    )
                    if start_ok and end_ok:
                        new_span = s1_row.copy()
                        new_span["id"] = str(uuid.uuid4())
                        new_span["geometry"] = LineString(cleaned)
                        new_span["start"] = s1_other["id"] if s1_other else None
                        new_span["end"] = s2_other["id"] if s2_other else None
                        new_span["name"] = _span_name_from_endpoints(
                            s1_other, s2_other
                        )
                        new_spans.append(new_span)

                spans_to_remove.extend([s1_idx, s2_idx])
                nodes_to_remove.add(node_a_id)
                nodes_to_remove.add(node_b_id)

        if spans_to_remove:
            gdf_ofds_spans = gdf_ofds_spans.drop(
                index=[i for i in spans_to_remove if i in gdf_ofds_spans.index]
            )
            if new_spans:
                new_gdf = gpd.GeoDataFrame(new_spans, crs=gdf_ofds_spans.crs)
                gdf_ofds_spans = pd.concat(
                    [gdf_ofds_spans, new_gdf], ignore_index=True
                )
        if nodes_to_remove:
            gdf_ofds_nodes = gdf_ofds_nodes[
                ~gdf_ofds_nodes["id"].isin(nodes_to_remove)
            ]
            print(
                f"Phase 5: Processed {len(unique_pairs)} pairs. "
                f"Removed {len(nodes_to_remove)} nodes, "
                f"joined {len(spans_to_remove)} spans into {len(new_spans)}. "
                f"Remaining nodes: {len(gdf_ofds_nodes)}"
            )

    return gdf_ofds_nodes, gdf_ofds_spans


def _split_spans_at_auto_generated_nodes(
    gdf_ofds_nodes,
    gdf_ofds_spans,
    threshold,
    threshold_meters,
    debug_enabled,
    debug_output_dir,
    debug_output_prefix,
):
    """Phase 6: Split spans at auto-generated nodes to create forks."""
    METERS_TO_DEGREES = 1.0 / 111000.0
    tol = ToleranceConfig.GEOMETRY_PRECISION

    start_ids = gdf_ofds_spans["start"].apply(_extract_id)
    end_ids = gdf_ofds_spans["end"].apply(_extract_id)
    span_endpoint_ids = set(pd.concat([start_ids, end_ids]).dropna())

    auto_endpoints = gdf_ofds_nodes[
        (gdf_ofds_nodes["name"] == AUTO_GENERATED_NODE_NAME)
        & (gdf_ofds_nodes["id"].isin(span_endpoint_ids))
    ]

    spans_to_remove = []
    new_spans = []
    nodes_to_rename = {}

    for node_idx, node_row in auto_endpoints.iterrows():
        node_point = node_row.geometry
        node_id = node_row["id"]

        node_span_idx = None
        node_is_start = False
        for span_idx, span_row in gdf_ofds_spans.iterrows():
            if span_idx in spans_to_remove:
                continue
            start_dict = parse_span_endpoint(span_row["start"], gdf_ofds_nodes)
            end_dict = parse_span_endpoint(span_row["end"], gdf_ofds_nodes)
            if start_dict and start_dict.get("id") == node_id:
                node_span_idx = span_idx
                node_is_start = True
                break
            elif end_dict and end_dict.get("id") == node_id:
                node_span_idx = span_idx
                node_is_start = False
                break

        if node_span_idx is None:
            continue

        min_dist = float("inf")
        nearest_span_idx = None
        nearest_pt = None
        for span_idx, span_row in gdf_ofds_spans.iterrows():
            if span_idx in spans_to_remove or span_idx == node_span_idx:
                continue
            pt = nearest_points(node_point, span_row.geometry)[1]
            d = node_point.distance(pt)
            if d < min_dist:
                min_dist = d
                nearest_span_idx = span_idx
                nearest_pt = pt

        if min_dist == float("inf") or min_dist > threshold:
            continue
        if nearest_span_idx is None or nearest_pt is None:
            continue

        target_row = gdf_ofds_spans.loc[nearest_span_idx]
        target_line = target_row.geometry
        if (
            nearest_pt.distance(Point(target_line.coords[0])) < tol
            or nearest_pt.distance(Point(target_line.coords[-1])) < tol
        ):
            continue

        node_span_row = gdf_ofds_spans.loc[node_span_idx]
        node_span_geom = node_span_row.geometry
        node_span_start = parse_span_endpoint(node_span_row["start"], gdf_ofds_nodes)
        node_span_end = parse_span_endpoint(node_span_row["end"], gdf_ofds_nodes)

        fork_info = {
            "id": node_id,
            "name": NETWORK_FORK_NAME,
            "location": {
                "type": "Point",
                "coordinates": [nearest_pt.x, nearest_pt.y],
            },
        }

        node_coords = list(node_span_geom.coords)
        fork_coord = (nearest_pt.x, nearest_pt.y)
        if node_is_start:
            if Point(node_coords[0]).distance(nearest_pt) > 1e-9:
                node_coords.insert(0, fork_coord)
            ext_start = fork_info
            ext_end = node_span_end.copy() if node_span_end else None
        else:
            if Point(node_coords[-1]).distance(nearest_pt) > 1e-9:
                node_coords.append(fork_coord)
            ext_start = node_span_start.copy() if node_span_start else None
            ext_end = fork_info

        ext_span = node_span_row.copy()
        ext_span["geometry"] = LineString(node_coords)
        ext_span["start"] = ext_start["id"] if ext_start else None
        ext_span["end"] = ext_end["id"] if ext_end else None
        ext_span["name"] = _span_name_from_endpoints(ext_start, ext_end)
        new_spans.append(ext_span)
        spans_to_remove.append(node_span_idx)

        target_coords = list(target_line.coords)
        split_coord = (nearest_pt.x, nearest_pt.y)
        insert_idx = None
        min_seg_dist = float("inf")
        for i in range(len(target_coords) - 1):
            seg = LineString([target_coords[i], target_coords[i + 1]])
            d = nearest_pt.distance(seg)
            if d < min_seg_dist:
                min_seg_dist = d
                if Point(target_coords[i]).distance(nearest_pt) < tol:
                    insert_idx = i
                elif Point(target_coords[i + 1]).distance(nearest_pt) < tol:
                    insert_idx = i + 1
                else:
                    insert_idx = i + 1

        if insert_idx is None:
            continue

        exists = any(
            Point(target_coords[j]).distance(nearest_pt) < tol
            for j in range(len(target_coords))
        )
        if not exists:
            target_coords.insert(insert_idx, split_coord)

        seg1_coords = target_coords[: insert_idx + 1]
        seg2_coords = target_coords[insert_idx:]
        if len(seg1_coords) < 2 or len(seg2_coords) < 2:
            continue

        seg1 = LineString(seg1_coords)
        seg2 = LineString(seg2_coords)
        target_start = parse_span_endpoint(target_row["start"], gdf_ofds_nodes)
        target_end = parse_span_endpoint(target_row["end"], gdf_ofds_nodes)

        seg1_start = Point(seg1.coords[0])
        seg1_end = Point(seg1.coords[-1])
        orig_start_pt = (
            Point(target_start["location"]["coordinates"])
            if target_start and target_start.get("location")
            else None
        )

        if orig_start_pt and seg1_start.distance(orig_start_pt) < 1e-3:
            s1_start, s1_end = target_start, fork_info
            s2_start, s2_end = fork_info, target_end
            s1_geom, s2_geom = seg1, seg2
        elif orig_start_pt and seg1_end.distance(orig_start_pt) < 1e-3:
            s1_start, s1_end = target_start, fork_info
            s2_start, s2_end = fork_info, target_end
            s1_geom, s2_geom = seg2, seg1
        else:
            s1_start, s1_end = target_start, fork_info
            s2_start, s2_end = fork_info, target_end
            s1_geom, s2_geom = seg2, seg1

        for s, start_d, end_d, geom in [
            (target_row, s1_start, s1_end, s1_geom),
            (target_row, s2_start, s2_end, s2_geom),
        ]:
            ns = s.copy()
            ns["id"] = str(uuid.uuid4())
            ns["geometry"] = geom
            ns["start"] = start_d["id"] if start_d else None
            ns["end"] = end_d["id"] if end_d else None
            ns["name"] = _span_name_from_endpoints(start_d, end_d)
            new_spans.append(ns)
        spans_to_remove.append(nearest_span_idx)
        nodes_to_rename[node_id] = nearest_pt

    if spans_to_remove:
        gdf_ofds_spans = gdf_ofds_spans.drop(
            index=[i for i in spans_to_remove if i in gdf_ofds_spans.index]
        )
        if new_spans:
            new_gdf = gpd.GeoDataFrame(new_spans, crs=gdf_ofds_spans.crs)
            gdf_ofds_spans = pd.concat(
                [gdf_ofds_spans, new_gdf], ignore_index=True
            )

    for node_id, new_geom in nodes_to_rename.items():
        match = gdf_ofds_nodes[gdf_ofds_nodes["id"] == node_id]
        if len(match) > 0:
            idx = match.index[0]
            gdf_ofds_nodes.at[idx, "geometry"] = new_geom
            gdf_ofds_nodes.at[idx, "name"] = NETWORK_FORK_NAME

    print(
        f"Phase 6: Moved and renamed {len(nodes_to_rename)} nodes to fork points, "
        f"split spans"
    )
    return gdf_ofds_nodes, gdf_ofds_spans


def _merge_network_forks_with_proper_nodes(
    gdf_ofds_nodes, gdf_ofds_spans, threshold
):
    """Phase 6b: Merge network fork nodes close to proper nodes."""
    fork_nodes = gdf_ofds_nodes[
        gdf_ofds_nodes["name"] == NETWORK_FORK_NAME
    ]
    if len(fork_nodes) == 0:
        return gdf_ofds_nodes, gdf_ofds_spans

    coords = np.array([(p.x, p.y) for p in gdf_ofds_nodes.geometry])
    tree = cKDTree(coords)
    clusters = [
        list(idxs) for idxs in tree.query_ball_point(coords, r=threshold)
        if len(idxs) > 1
    ]

    fork_to_proper = {}
    for cluster in clusters:
        fork_indices = []
        proper_indices = []
        for idx in cluster:
            node_name = gdf_ofds_nodes.iloc[idx]["name"]
            if node_name == NETWORK_FORK_NAME:
                fork_indices.append(idx)
            elif node_name != AUTO_GENERATED_NODE_NAME:
                proper_indices.append(idx)
        if fork_indices and proper_indices:
            proper_id = gdf_ofds_nodes.iloc[proper_indices[0]]["id"]
            for fork_idx in fork_indices:
                fork_to_proper[gdf_ofds_nodes.iloc[fork_idx]["id"]] = proper_id

    if not fork_to_proper:
        return gdf_ofds_nodes, gdf_ofds_spans

    merged_ids = []
    for index, span in gdf_ofds_spans.iterrows():
        start_dict = parse_span_endpoint(span["start"], gdf_ofds_nodes)
        end_dict = parse_span_endpoint(span["end"], gdf_ofds_nodes)
        start_updated = False
        end_updated = False

        if start_dict and (start_id := start_dict.get("id")) in fork_to_proper:
            proper_id = fork_to_proper[start_id]
            proper_row = gdf_ofds_nodes[gdf_ofds_nodes["id"] == proper_id]
            if not proper_row.empty:
                p = proper_row.iloc[0]
                start_dict["id"] = p["id"]
                start_dict["name"] = p["name"]
                if "location" in start_dict:
                    start_dict["location"]["coordinates"] = [p.geometry.x, p.geometry.y]
                merged_ids.append(start_id)
                start_updated = True
                geom = span["geometry"]
                coords = list(geom.coords)
                coords[0] = (p.geometry.x, p.geometry.y)
                gdf_ofds_spans.at[index, "geometry"] = LineString(coords)

        if end_dict and (end_id := end_dict.get("id")) in fork_to_proper:
            proper_id = fork_to_proper[end_id]
            proper_row = gdf_ofds_nodes[gdf_ofds_nodes["id"] == proper_id]
            if not proper_row.empty:
                p = proper_row.iloc[0]
                end_dict["id"] = p["id"]
                end_dict["name"] = p["name"]
                if "location" in end_dict:
                    end_dict["location"]["coordinates"] = [p.geometry.x, p.geometry.y]
                merged_ids.append(end_id)
                end_updated = True
                geom = gdf_ofds_spans.at[index, "geometry"]
                coords = list(geom.coords)
                coords[-1] = (p.geometry.x, p.geometry.y)
                gdf_ofds_spans.at[index, "geometry"] = LineString(coords)

        if start_updated or end_updated:
            gdf_ofds_spans.at[index, "start"] = (
                start_dict["id"] if start_dict else None
            )
            gdf_ofds_spans.at[index, "end"] = (
                end_dict["id"] if end_dict else None
            )
            gdf_ofds_spans.at[index, "name"] = _span_name_from_endpoints(
                start_dict, end_dict
            )

    gdf_ofds_nodes = gdf_ofds_nodes[~gdf_ofds_nodes["id"].isin(merged_ids)]
    print(
        f"Phase 6b: Merged {len(set(merged_ids))} network fork nodes "
        f"with proper nodes. Remaining nodes: {len(gdf_ofds_nodes)}"
    )
    return gdf_ofds_nodes, gdf_ofds_spans


def _split_spans_at_proper_nodes(gdf_ofds_nodes, gdf_ofds_spans, threshold):
    """Phase 7: Split spans at proper nodes near but not on spans."""
    tol = ToleranceConfig.GEOMETRY_PRECISION

    start_ids = gdf_ofds_spans["start"].apply(_extract_id)
    end_ids = gdf_ofds_spans["end"].apply(_extract_id)
    span_endpoint_ids = set(pd.concat([start_ids, end_ids]).dropna())

    proper = gdf_ofds_nodes[
        (gdf_ofds_nodes["name"] != AUTO_GENERATED_NODE_NAME)
        & (gdf_ofds_nodes["name"] != NETWORK_FORK_NAME)
        & (~gdf_ofds_nodes["id"].isin(span_endpoint_ids))
    ]

    spans_to_remove = []
    new_spans = []

    for node_idx, node_row in proper.iterrows():
        node_point = node_row.geometry
        node_id = node_row["id"]
        node_name = node_row["name"]

        current_span_ids = []
        for span_idx, span_row in gdf_ofds_spans.iterrows():
            if span_idx in spans_to_remove:
                continue
            start_dict = parse_span_endpoint(span_row["start"], gdf_ofds_nodes)
            end_dict = parse_span_endpoint(span_row["end"], gdf_ofds_nodes)
            if (
                (start_dict and start_dict.get("id") == node_id)
                or (end_dict and end_dict.get("id") == node_id)
            ):
                current_span_ids.append(span_idx)

        min_dist = float("inf")
        nearest_span_idx = None
        nearest_pt = None
        for span_idx, span_row in gdf_ofds_spans.iterrows():
            if span_idx in spans_to_remove or span_idx in current_span_ids:
                continue
            pt = nearest_points(node_point, span_row.geometry)[1]
            d = node_point.distance(pt)
            if d < min_dist:
                min_dist = d
                nearest_span_idx = span_idx
                nearest_pt = pt

        if (
            min_dist > threshold
            or nearest_span_idx is None
            or nearest_pt is None
        ):
            continue

        span_row = gdf_ofds_spans.loc[nearest_span_idx]
        span_line = span_row.geometry
        if (
            nearest_pt.distance(Point(span_line.coords[0])) < tol
            or nearest_pt.distance(Point(span_line.coords[-1])) < tol
        ):
            continue

        coords = list(span_line.coords)
        split_coord = (nearest_pt.x, nearest_pt.y)
        insert_idx = None
        for i in range(len(coords) - 1):
            seg = LineString([coords[i], coords[i + 1]])
            d = nearest_pt.distance(seg)
            if Point(coords[i]).distance(nearest_pt) < 1e-9:
                insert_idx = i + 1
                break
            elif Point(coords[i + 1]).distance(nearest_pt) < 1e-9:
                insert_idx = i + 1
                break
            else:
                insert_idx = i + 1

        if insert_idx is None:
            continue

        exists = any(Point(c).distance(nearest_pt) < 1e-9 for c in coords)
        if not exists:
            coords.insert(insert_idx, split_coord)

        seg1_coords = coords[: insert_idx + 1]
        seg2_coords = coords[insert_idx:]
        if len(seg1_coords) < 2 or len(seg2_coords) < 2:
            continue

        seg1 = LineString(seg1_coords)
        seg2 = LineString(seg2_coords)

        if Point(seg1.coords[-1]).distance(node_point) > 1e-9:
            seg1 = LineString(
                list(seg1.coords) + [(node_point.x, node_point.y)]
            )
        if Point(seg2.coords[0]).distance(node_point) > 1e-9:
            seg2 = LineString(
                [(node_point.x, node_point.y)] + list(seg2.coords)
            )

        orig_start = parse_span_endpoint(span_row["start"], gdf_ofds_nodes)
        orig_end = parse_span_endpoint(span_row["end"], gdf_ofds_nodes)
        proper_info = {
            "id": node_id,
            "name": node_name,
            "location": {
                "type": "Point",
                "coordinates": [node_point.x, node_point.y],
            },
        }

        seg1_start = Point(seg1.coords[0])
        seg2_start = Point(seg2.coords[0])
        orig_start_pt = (
            Point(orig_start["location"]["coordinates"])
            if orig_start and orig_start.get("location")
            else None
        )

        if orig_start_pt and seg1_start.distance(orig_start_pt) < 1e-3:
            s1_start, s1_end = orig_start, proper_info
            s2_start, s2_end = proper_info, orig_end
            s1_geom, s2_geom = seg1, seg2
        elif orig_start_pt and seg2_start.distance(orig_start_pt) < 1e-3:
            s1_start, s1_end = orig_start, proper_info
            s2_start, s2_end = proper_info, orig_end
            s1_geom, s2_geom = seg2, seg1
        else:
            orig_end_pt = (
                Point(orig_end["location"]["coordinates"])
                if orig_end and orig_end.get("location")
                else None
            )
            if orig_end_pt and Point(seg1.coords[-1]).distance(orig_end_pt) < 1e-3:
                s1_start, s1_end = orig_start, proper_info
                s2_start, s2_end = proper_info, orig_end
                s1_geom, s2_geom = seg2, seg1
            else:
                s1_start, s1_end = orig_start, proper_info
                s2_start, s2_end = proper_info, orig_end
                s1_geom, s2_geom = seg1, seg2

        for start_d, end_d, geom in [(s1_start, s1_end, s1_geom), (s2_start, s2_end, s2_geom)]:
            ns = span_row.copy()
            ns["id"] = str(uuid.uuid4())
            ns["geometry"] = geom
            ns["start"] = start_d["id"] if start_d else None
            ns["end"] = end_d["id"] if end_d else None
            ns["name"] = _span_name_from_endpoints(start_d, end_d)
            new_spans.append(ns)
        spans_to_remove.append(nearest_span_idx)

    if spans_to_remove:
        gdf_ofds_spans = gdf_ofds_spans.drop(
            index=[i for i in spans_to_remove if i in gdf_ofds_spans.index]
        )
        if new_spans:
            new_gdf = gpd.GeoDataFrame(new_spans, crs=gdf_ofds_spans.crs)
            gdf_ofds_spans = pd.concat(
                [gdf_ofds_spans, new_gdf], ignore_index=True
            )
        print(
            f"Phase 7: Processed {len(proper)} proper nodes, "
            f"split {len(spans_to_remove)} spans"
        )

    return gdf_ofds_nodes, gdf_ofds_spans


def _split_spans_at_proper_node_endpoints(
    gdf_ofds_nodes, gdf_ofds_spans, threshold
):
    """Phase 7b: Split spans at proper nodes that are already span endpoints."""
    tol = ToleranceConfig.GEOMETRY_PRECISION

    start_ids = gdf_ofds_spans["start"].apply(_extract_id)
    end_ids = gdf_ofds_spans["end"].apply(_extract_id)
    span_endpoint_ids = set(pd.concat([start_ids, end_ids]).dropna())

    proper_endpoints = gdf_ofds_nodes[
        (gdf_ofds_nodes["name"] != AUTO_GENERATED_NODE_NAME)
        & (gdf_ofds_nodes["name"] != NETWORK_FORK_NAME)
        & (gdf_ofds_nodes["id"].isin(span_endpoint_ids))
    ]

    spans_to_remove = []
    new_spans = []
    splits_count = 0

    for node_idx, node_row in proper_endpoints.iterrows():
        node_point = node_row.geometry
        node_id = node_row["id"]
        node_name = node_row["name"]

        current_span_ids = []
        for span_idx, span_row in gdf_ofds_spans.iterrows():
            if span_idx in spans_to_remove:
                continue
            start_dict = parse_span_endpoint(span_row["start"], gdf_ofds_nodes)
            end_dict = parse_span_endpoint(span_row["end"], gdf_ofds_nodes)
            if (
                (start_dict and start_dict.get("id") == node_id)
                or (end_dict and end_dict.get("id") == node_id)
            ):
                current_span_ids.append(span_idx)

        min_dist = float("inf")
        nearest_span_idx = None
        nearest_pt = None
        for span_idx, span_row in gdf_ofds_spans.iterrows():
            if span_idx in spans_to_remove or span_idx in current_span_ids:
                continue
            pt = nearest_points(node_point, span_row.geometry)[1]
            d = node_point.distance(pt)
            if d < min_dist:
                min_dist = d
                nearest_span_idx = span_idx
                nearest_pt = pt

        if (
            min_dist > threshold
            or nearest_span_idx is None
            or nearest_pt is None
        ):
            continue

        span_row = gdf_ofds_spans.loc[nearest_span_idx]
        span_line = span_row.geometry
        if (
            nearest_pt.distance(Point(span_line.coords[0])) < tol
            or nearest_pt.distance(Point(span_line.coords[-1])) < tol
        ):
            continue

        coords = list(span_line.coords)
        split_coord = (node_point.x, node_point.y)
        insert_idx = None
        for i in range(len(coords) - 1):
            seg = LineString([coords[i], coords[i + 1]])
            d = nearest_pt.distance(seg)
            if Point(coords[i]).distance(nearest_pt) < 1e-9:
                insert_idx = i + 1
                break
            elif Point(coords[i + 1]).distance(nearest_pt) < 1e-9:
                insert_idx = i + 1
                break
            else:
                insert_idx = i + 1

        if insert_idx is None:
            continue

        exists = any(Point(c).distance(nearest_pt) < 1e-9 for c in coords)
        if not exists:
            coords.insert(insert_idx, split_coord)

        seg1_coords = list(coords[: insert_idx + 1])
        seg2_coords = list(coords[insert_idx:])
        if len(seg1_coords) < 2 or len(seg2_coords) < 2:
            continue

        seg1_coords[-1] = split_coord
        seg2_coords[0] = split_coord
        seg1 = LineString(seg1_coords)
        seg2 = LineString(seg2_coords)

        orig_start = parse_span_endpoint(span_row["start"], gdf_ofds_nodes)
        orig_end = parse_span_endpoint(span_row["end"], gdf_ofds_nodes)
        proper_info = {
            "id": node_id,
            "name": node_name,
            "location": {
                "type": "Point",
                "coordinates": [node_point.x, node_point.y],
            },
        }

        seg1_start = Point(seg1.coords[0])
        seg2_start = Point(seg2.coords[0])
        orig_start_pt = (
            Point(orig_start["location"]["coordinates"])
            if orig_start and orig_start.get("location")
            else None
        )

        if orig_start_pt and seg1_start.distance(orig_start_pt) < 1e-3:
            s1_start, s1_end = orig_start, proper_info
            s2_start, s2_end = proper_info, orig_end
            s1_geom, s2_geom = seg1, seg2
        elif orig_start_pt and seg2_start.distance(orig_start_pt) < 1e-3:
            s1_start, s1_end = orig_start, proper_info
            s2_start, s2_end = proper_info, orig_end
            s1_geom, s2_geom = seg2, seg1
        else:
            orig_end_pt = (
                Point(orig_end["location"]["coordinates"])
                if orig_end and orig_end.get("location")
                else None
            )
            if orig_end_pt and Point(seg1.coords[-1]).distance(orig_end_pt) < 1e-3:
                s1_start, s1_end = orig_start, proper_info
                s2_start, s2_end = proper_info, orig_end
                s1_geom, s2_geom = seg2, seg1
            else:
                s1_start, s1_end = orig_start, proper_info
                s2_start, s2_end = proper_info, orig_end
                s1_geom, s2_geom = seg1, seg2

        for start_d, end_d, geom in [
            (s1_start, s1_end, s1_geom),
            (s2_start, s2_end, s2_geom),
        ]:
            ns = span_row.copy()
            ns["id"] = str(uuid.uuid4())
            ns["geometry"] = geom
            ns["start"] = start_d["id"] if start_d else None
            ns["end"] = end_d["id"] if end_d else None
            ns["name"] = _span_name_from_endpoints(start_d, end_d)
            new_spans.append(ns)
        spans_to_remove.append(nearest_span_idx)
        splits_count += 1

    if spans_to_remove:
        gdf_ofds_spans = gdf_ofds_spans.drop(
            index=[i for i in spans_to_remove if i in gdf_ofds_spans.index]
        )
        if new_spans:
            new_gdf = gpd.GeoDataFrame(new_spans, crs=gdf_ofds_spans.crs)
            gdf_ofds_spans = pd.concat(
                [gdf_ofds_spans, new_gdf], ignore_index=True
            )
        print(
            f"Phase 7b: Split {splits_count} spans at proper node endpoints"
        )

    return gdf_ofds_nodes, gdf_ofds_spans


def _remove_duplicate_spans(gdf_ofds_spans):
    """Phase 8: Remove bidirectional duplicate spans."""
    print("\nPhase 8: Removing bidirectional duplicate spans...")
    before = len(gdf_ofds_spans)
    normalized = {}
    to_keep = []

    for index, span in gdf_ofds_spans.iterrows():
        start_dict = parse_span_endpoint(span["start"])
        end_dict = parse_span_endpoint(span["end"])
        start_id = start_dict.get("id") if start_dict else None
        end_id = end_dict.get("id") if end_dict else None

        if start_id is None or end_id is None:
            to_keep.append(index)
            continue

        key = tuple(sorted([start_id, end_id]))
        if key not in normalized:
            normalized[key] = index
            to_keep.append(index)
        else:
            existing = gdf_ofds_spans.loc[normalized[key]]
            ex_coords = list(existing.geometry.coords)
            cur_coords = list(span.geometry.coords)
            if ex_coords == cur_coords or ex_coords == cur_coords[::-1]:
                continue
            continue

    gdf_ofds_spans = gdf_ofds_spans.loc[to_keep].copy()
    removed = before - len(gdf_ofds_spans)
    if removed > 0:
        print(f"Phase 8: Removed {removed} bidirectional duplicate spans")
    return gdf_ofds_spans


def _rename_spans_from_nodes(gdf_ofds_nodes, gdf_ofds_spans):
    """Phase 9: Set span names from start/end node names for all spans."""
    print("\nPhase 9: Setting span names from node names...")
    count = 0
    for index, span in gdf_ofds_spans.iterrows():
        start_dict = parse_span_endpoint(span["start"], gdf_ofds_nodes)
        end_dict = parse_span_endpoint(span["end"], gdf_ofds_nodes)
        name = _span_name_from_endpoints(start_dict, end_dict)
        gdf_ofds_spans.at[index, "name"] = name
        count += 1
    print(f"Phase 9: Set names for {count} spans")
    return gdf_ofds_spans


def _validate_node_references(gdf_ofds_nodes, gdf_ofds_spans):
    """Phase 10: Validate span node references exist in nodes."""
    print("\nPhase 10: Validating node references...")
    valid_ids = set(gdf_ofds_nodes["id"].unique())
    invalid = []
    for index, span in gdf_ofds_spans.iterrows():
        start_dict = parse_span_endpoint(span["start"], gdf_ofds_nodes)
        end_dict = parse_span_endpoint(span["end"], gdf_ofds_nodes)
        start_id = start_dict.get("id") if start_dict else None
        end_id = end_dict.get("id") if end_dict else None
        issues = []
        if start_id and start_id not in valid_ids:
            issues.append(f"start node ID '{start_id}' not found")
        if end_id and end_id not in valid_ids:
            issues.append(f"end node ID '{end_id}' not found")
        if issues:
            invalid.append({"name": span.get("name", "?"), "issues": issues})
    if invalid:
        print(f"WARNING: {len(invalid)} spans with invalid node references")
    else:
        print("Phase 10: All spans have valid node references")


def consolidate_auto_generated_nodes(
    gdf_ofds_nodes,
    gdf_ofds_spans,
    threshold_meters,
    debug_enabled=False,
    debug_output_dir=None,
    debug_output_prefix="",
    rename_spans_from_nodes=True,
):
    """Analyze, merge, split, and clean auto-generated nodes."""
    setup = _setup_consolidation(
        gdf_ofds_nodes, gdf_ofds_spans, threshold_meters
    )
    if setup is None:
        return gdf_ofds_spans, gdf_ofds_nodes

    threshold, auto_gen_nodes, other_nodes, span_endpoint_ids = setup
    _analyze_auto_generated_nodes(
        auto_gen_nodes, other_nodes, gdf_ofds_spans, span_endpoint_ids
    )

    if debug_enabled and debug_output_dir:
        write_debug_geojson(
            gdf_ofds_nodes, gdf_ofds_spans,
            debug_output_dir, "phase2", debug_output_prefix
        )

    gdf_ofds_nodes, gdf_ofds_spans = _merge_auto_generated_with_proper_nodes(
        gdf_ofds_nodes, gdf_ofds_spans, threshold
    )
    if debug_enabled and debug_output_dir:
        write_debug_geojson(
            gdf_ofds_nodes, gdf_ofds_spans,
            debug_output_dir, "phase3", debug_output_prefix
        )

    gdf_ofds_nodes, gdf_ofds_spans = _merge_large_auto_generated_clusters(
        gdf_ofds_nodes, gdf_ofds_spans, threshold
    )
    if debug_enabled and debug_output_dir:
        write_debug_geojson(
            gdf_ofds_nodes, gdf_ofds_spans,
            debug_output_dir, "phase4", debug_output_prefix
        )

    gdf_ofds_nodes, gdf_ofds_spans = _merge_pair_auto_generated_nodes(
        gdf_ofds_nodes, gdf_ofds_spans, threshold
    )
    if debug_enabled and debug_output_dir:
        write_debug_geojson(
            gdf_ofds_nodes, gdf_ofds_spans,
            debug_output_dir, "phase5", debug_output_prefix
        )

    gdf_ofds_nodes, gdf_ofds_spans = _split_spans_at_auto_generated_nodes(
        gdf_ofds_nodes,
        gdf_ofds_spans,
        threshold,
        threshold_meters,
        debug_enabled,
        debug_output_dir or "",
        debug_output_prefix,
    )
    if debug_enabled and debug_output_dir:
        write_debug_geojson(
            gdf_ofds_nodes, gdf_ofds_spans,
            debug_output_dir, "phase6", debug_output_prefix
        )

    gdf_ofds_nodes, gdf_ofds_spans = _merge_network_forks_with_proper_nodes(
        gdf_ofds_nodes, gdf_ofds_spans, threshold
    )
    if debug_enabled and debug_output_dir:
        write_debug_geojson(
            gdf_ofds_nodes, gdf_ofds_spans,
            debug_output_dir, "phase6b", debug_output_prefix
        )

    gdf_ofds_nodes, gdf_ofds_spans = _split_spans_at_proper_nodes(
        gdf_ofds_nodes, gdf_ofds_spans, threshold
    )
    if debug_enabled and debug_output_dir:
        write_debug_geojson(
            gdf_ofds_nodes, gdf_ofds_spans,
            debug_output_dir, "phase7", debug_output_prefix
        )

    gdf_ofds_nodes, gdf_ofds_spans = _split_spans_at_proper_node_endpoints(
        gdf_ofds_nodes, gdf_ofds_spans, threshold
    )
    if debug_enabled and debug_output_dir:
        write_debug_geojson(
            gdf_ofds_nodes, gdf_ofds_spans,
            debug_output_dir, "phase7b", debug_output_prefix
        )

    gdf_ofds_spans = _remove_duplicate_spans(gdf_ofds_spans)
    if debug_enabled and debug_output_dir:
        write_debug_geojson(
            gdf_ofds_nodes, gdf_ofds_spans,
            debug_output_dir, "phase8", debug_output_prefix
        )

    gdf_ofds_spans = _rename_spans_from_nodes(gdf_ofds_nodes, gdf_ofds_spans)
    if debug_enabled and debug_output_dir:
        write_debug_geojson(
            gdf_ofds_nodes, gdf_ofds_spans,
            debug_output_dir, "phase9", debug_output_prefix
        )

    _validate_node_references(gdf_ofds_nodes, gdf_ofds_spans)

    print(f"\nFinal counts: {len(gdf_ofds_spans)} spans, {len(gdf_ofds_nodes)} nodes")
    if debug_enabled and debug_output_dir:
        write_debug_geojson(
            gdf_ofds_nodes, gdf_ofds_spans,
            debug_output_dir, "phase11", debug_output_prefix
        )

    return gdf_ofds_spans, gdf_ofds_nodes
