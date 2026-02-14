"""
Constants for kml2ofds: tolerances, OFDS schema URLs, and feature types.
"""

# Default OFDS schema URL (0.3.0)
DEFAULT_NETWORK_LINKS = (
    "https://raw.githubusercontent.com/Open-Telecoms-Data/"
    "open-fibre-data-standard/0__3__0/schema/network-schema.json"
)

# Feature type strings for OFDS GeoJSON
FEATURE_TYPE_NODE = "node"
FEATURE_TYPE_SPAN = "span"

# Node type names
AUTO_GENERATED_NODE_NAME = "Auto generated missing node"
NETWORK_FORK_NAME = "network fork"


class ToleranceConfig:
    """Tolerance values for geometry and node matching operations."""

    # Node snap: distance threshold for snapping a node to the nearest span
    NODE_SNAP = 1e-4

    # Endpoint match: tolerance for matching span endpoints to nodes
    ENDPOINT_MATCH = 1e-3

    # Geometry precision for buffer/within operations
    GEOMETRY_PRECISION = 1e-6

    # Buffer size for very small tolerance operations
    BUFFER_SIZE = 1e-9
