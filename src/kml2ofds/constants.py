"""
Constants for kml2ofds: tolerances, OFDS schema URLs, and feature types.
"""

# Default OFDS schema URL (0.4.0)
DEFAULT_NETWORK_LINKS = (
    "https://standard.ofds.info/en/0__4__0/network-schema.json"
)

# Feature type strings for OFDS GeoJSON
FEATURE_TYPE_NODE = "node"
FEATURE_TYPE_SPAN = "span"

# Node type names
AUTO_GENERATED_NODE_NAME = "default-node"
NETWORK_FORK_NAME = "fork"


class ToleranceConfig:
    """Tolerance values for geometry and node matching operations."""

    # Node snap: distance threshold for snapping a node to the nearest span
    NODE_SNAP = 1e-4

    # Endpoint match: tolerance for matching span endpoints to nodes
    ENDPOINT_MATCH = 1e-3

    # Geometry precision for buffer/within operations
    GEOMETRY_PRECISION = 1e-6

    # break_spans_at_node_points: max distance (planar degrees in CRS) from node to
    # line for intersection + split. Same as NODE_SNAP; 1e-9° was below typical KML
    # precision and missed breaks.
    BUFFER_SIZE = NODE_SNAP
