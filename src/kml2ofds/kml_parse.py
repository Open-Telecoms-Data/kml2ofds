"""
KML parsing: extract nodes and spans from KML files into GeoDataFrames.
"""

import re
import sys
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pykml import parser
from shapely.geometry import Point, LineString
import geopandas as gpd

from .constants import DEFAULT_NETWORK_LINKS, FEATURE_TYPE_NODE, FEATURE_TYPE_SPAN

if TYPE_CHECKING:
    from .config import Config


@dataclass
class ParseContext:
    """Context passed to placemark processing (from Config)."""

    network_id: str
    network_name: str
    network_links: list[dict]
    physical_infrastructure_provider_id: str
    physical_infrastructure_provider_name: str
    network_providers_id: str
    network_providers_name: str
    ignore_placemarks: list[str]


def _network_info(ctx: ParseContext) -> dict:
    """Build network info dict for OFDS features."""
    return {
        "id": ctx.network_id,
        "name": ctx.network_name,
        "links": ctx.network_links
        if ctx.network_links
        else [{"rel": "describedby", "href": DEFAULT_NETWORK_LINKS}],
    }


def _create_geojson_node(
    point: Point,
    name: str,
    ctx: ParseContext,
) -> dict:
    """Create a GeoJSON node feature from a point and context."""
    node_id = str(uuid.uuid4())
    return {
        "type": "Feature",
        "properties": {
            "name": name,
            "id": node_id,
            "network": _network_info(ctx),
            "physicalInfrastructureProvider": {
                "id": ctx.physical_infrastructure_provider_id,
                "name": ctx.physical_infrastructure_provider_name,
            },
            "networkProviders": [
                {
                    "id": ctx.network_providers_id,
                    "name": ctx.network_providers_name,
                }
            ],
            "featureType": FEATURE_TYPE_NODE,
        },
        "geometry": {
            "type": "Point",
            "coordinates": [point.x, point.y],
        },
    }


def _create_geojson_span(
    linestring: LineString,
    name: str,
    ctx: ParseContext,
) -> dict:
    """Create a GeoJSON span feature from a LineString and context."""
    return {
        "type": "Feature",
        "properties": {
            "id": "",
            "name": name,
            "network": _network_info(ctx),
            "physicalInfrastructureProvider": {
                "id": ctx.physical_infrastructure_provider_id,
                "name": ctx.physical_infrastructure_provider_name,
            },
            "networkProviders": [
                {
                    "id": ctx.network_providers_id,
                    "name": ctx.network_providers_name,
                }
            ],
            "featureType": FEATURE_TYPE_SPAN,
        },
        "geometry": {
            "type": "LineString",
            "coordinates": [(x, y) for x, y, *_ in linestring.coords],
        },
    }


def _is_placemark_ignored(name: str, ignore_patterns: list[str]) -> bool:
    """Check if a placemark name matches any ignore pattern."""
    for pattern in ignore_patterns:
        if re.search(rf"{pattern}", name):
            return True
    return False


def remove_duplicate_nodes(geojson_nodes: list[dict], precision: int = 1) -> list[dict]:
    """Remove duplicate nodes by name and rounded coordinates."""
    unique_nodes = []
    seen_hashes = set()
    for node in geojson_nodes:
        node_hash = hash(
            (
                node["properties"]["name"],
                round(node["geometry"]["coordinates"][0], precision),
                round(node["geometry"]["coordinates"][1], precision),
            )
        )
        if node_hash not in seen_hashes:
            unique_nodes.append(node)
            seen_hashes.add(node_hash)
    return unique_nodes


def _process_placemark(
    placemark,
    ctx: ParseContext,
    geojson_nodes: list,
    geojson_spans: list,
) -> None:
    """Process a single Placemark and append nodes/spans to the lists."""
    KML_NS = "{http://www.opengis.net/kml/2.2}"

    name_elem = placemark.find(f"{KML_NS}name")
    name = name_elem.text if name_elem is not None else "Default Name"

    # Point geometry
    point_geom = placemark.find(f"{KML_NS}Point")
    if point_geom is not None:
        coords_elem = point_geom.find(f"{KML_NS}coordinates")
        if coords_elem is not None:
            parts = coords_elem.text.split(",")
            shapely_point = Point(float(parts[0]), float(parts[1]))
            if not _is_placemark_ignored(name, ctx.ignore_placemarks):
                geojson_nodes.append(_create_geojson_node(shapely_point, name, ctx))

    # MultiGeometry
    multi_geometry = placemark.find(f"{KML_NS}MultiGeometry")
    if multi_geometry is not None:
        for point_elem in multi_geometry.iter(f"{KML_NS}Point"):
            coords_elem = point_elem.find(f"{KML_NS}coordinates")
            if coords_elem is not None:
                coords = tuple(map(float, coords_elem.text.split(",")[:2]))
                if not _is_placemark_ignored(name, ctx.ignore_placemarks):
                    geojson_nodes.append(
                        _create_geojson_node(Point(coords[0], coords[1]), name, ctx)
                    )

        for line_elem in multi_geometry.iter(f"{KML_NS}LineString"):
            coords_elem = line_elem.find(f"{KML_NS}coordinates")
            if coords_elem is not None:
                coordinates = [
                    tuple(map(float, coord.split(",")))
                    for coord in coords_elem.text.split()
                ]
                if len(coordinates) >= 2:
                    shapely_line = LineString(coordinates)
                    geojson_span = _create_geojson_span(shapely_line, name, ctx)
                    is_dup = any(
                        s["properties"]["name"] == name
                        and s["geometry"]["coordinates"]
                        == geojson_span["geometry"]["coordinates"]
                        for s in geojson_spans
                    )
                    if not is_dup:
                        geojson_spans.append(geojson_span)

    # Standalone LineString
    elif placemark.find(f"{KML_NS}LineString") is not None:
        polyline = placemark.find(f"{KML_NS}LineString")
        if polyline is not None:
            coords_elem = polyline.find(f"{KML_NS}coordinates")
            if coords_elem is not None:
                coordinates = [
                    tuple(map(float, coord.split(",")))
                    for coord in coords_elem.text.split()
                ]
                if len(coordinates) >= 2:
                    shapely_line = LineString(coordinates)
                    geojson_span = _create_geojson_span(shapely_line, name, ctx)
                    is_dup = any(
                        s["properties"]["name"] == name
                        and s["geometry"]["coordinates"]
                        == geojson_span["geometry"]["coordinates"]
                        for s in geojson_spans
                    )
                    if not is_dup:
                        geojson_spans.append(geojson_span)


def _process_document_element(document, ctx: ParseContext) -> tuple[list[dict], list[dict]]:
    """Process a KML Document or Folder and return (nodes, spans)."""
    KML_NS = "{http://www.opengis.net/kml/2.2}"
    geojson_nodes = []
    geojson_spans = []

    for placemark in document.findall(f"{KML_NS}Placemark"):
        _process_placemark(placemark, ctx, geojson_nodes, geojson_spans)

    for folder in document.iter(f"{KML_NS}Folder"):
        for placemark in folder.iter(f"{KML_NS}Placemark"):
            _process_placemark(placemark, ctx, geojson_nodes, geojson_spans)

    for nested_doc in document.findall(f"{KML_NS}Document"):
        nested_name = nested_doc.findtext(f"{KML_NS}name")
        print(f"Processing nested Document: {nested_name}")
        nested_nodes, nested_spans = _process_document_element(nested_doc, ctx)
        geojson_nodes.extend(nested_nodes)
        geojson_spans.extend(nested_spans)

    return geojson_nodes, geojson_spans


def _ctx_from_config(config: "Config") -> ParseContext:
    """Build ParseContext from Config."""
    return ParseContext(
        network_id=config.network_id,
        network_name=config.network_name,
        network_links=config.network_links,
        physical_infrastructure_provider_id=config.physical_infrastructure_provider_id,
        physical_infrastructure_provider_name=config.physical_infrastructure_provider_name,
        network_providers_id=config.network_providers_id,
        network_providers_name=config.network_providers_name,
        ignore_placemarks=config.ignore_placemarks,
    )


def process_kml_file(path: str, config: "Config") -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Parse KML file and return (nodes_gdf, spans_gdf). No file writes."""
    try:
        with open(path) as f:
            kml_doc = parser.parse(f).getroot()
    except FileNotFoundError:
        print(f"\nERROR: KML file not found: {path}")
        sys.exit(1)
    except PermissionError:
        print(f"\nERROR: Permission denied when trying to read KML file: {path}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Failed to read or parse KML file: {path}")
        print(f"  Error: {type(e).__name__}: {str(e)}")
        sys.exit(1)

    geojson_nodes = []
    geojson_spans = []
    ctx = _ctx_from_config(config)
    KML_NS = "{http://www.opengis.net/kml/2.2}"

    all_documents = list(kml_doc.iter(f"{KML_NS}Document"))

    if all_documents:
        for document in all_documents:
            is_nested = False
            parent = document.getparent()
            while parent is not None and parent != kml_doc:
                if parent.tag == f"{KML_NS}Document":
                    is_nested = True
                    break
                parent = parent.getparent()

            if is_nested:
                continue

            doc_name = document.findtext(f"{KML_NS}name")
            print(f"Processing Document: {doc_name}", flush=True)
            nodes, spans = _process_document_element(document, ctx)
            geojson_nodes.extend(nodes)
            geojson_spans.extend(spans)
    else:
        root_folders = [
            c for c in kml_doc
            if c.tag == f"{KML_NS}Folder"
        ]
        if not root_folders:
            if kml_doc.tag == f"{KML_NS}Folder":
                root_folders = [kml_doc]
            else:
                root_folders = list(kml_doc.iter(f"{KML_NS}Folder"))

        for folder in root_folders:
            folder_name = folder.findtext(f"{KML_NS}name")
            print(f"Processing Folder: {folder_name}")
            nodes, spans = _process_document_element(folder, ctx)
            geojson_nodes.extend(nodes)
            geojson_spans.extend(spans)

    print(f"Number of nodes found before deduplication: {len(geojson_nodes)}", flush=True)
    geojson_nodes = remove_duplicate_nodes(geojson_nodes, 1)
    print(f"Number of nodes found after deduplication: {len(geojson_nodes)}", flush=True)

    gdf_nodes = gpd.GeoDataFrame.from_features(geojson_nodes)
    gdf_spans = gpd.GeoDataFrame.from_features(geojson_spans)

    # Snap nodes to spans (import here to avoid circular import at module load)
    from .geometry import snap_to_line

    n_nodes = len(gdf_nodes)
    n_spans = len(gdf_spans)
    print(
        f"Snapping {n_nodes} nodes to {n_spans} spans "
        f"(this may take several minutes for large files)...",
        flush=True,
    )
    snapped = gdf_nodes.geometry.map(
        lambda p: snap_to_line(p, gdf_spans)
    )
    gdf_ofds_nodes = gpd.GeoDataFrame(gdf_nodes.drop(columns="geometry").copy())
    gdf_ofds_nodes["geometry"] = snapped
    gdf_ofds_nodes.set_geometry("geometry", inplace=True)

    return gdf_ofds_nodes, gdf_spans
