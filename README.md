# KML2OFDS

KML2OFDS is a Python script for converting KML maps of fibre optic network infrastructure to the [Open Fibre Data Standard][ofds-repo]. Consult the [documentation][ofds-docs] for more info.

Requires Python 3.10+.

## Usage

KML2OFDS is driven by a **network profile** (INI configuration file). You must provide the path to your profile:

```sh
kml2ofds --network-profile path/to/your/profile.ini
```

The profile specifies the KML file to convert, input/output directories, network metadata, and processing options. See `default.profile` for a full example.

## Assumptions

Core to OFDS is the principle that any span of fibre must be terminated at either end by a Point of Presence of some kind. Here we use Point of Presence in a loose sense—it might be a simple access point such as a manhole or a full point of presence. Consult the standard for more details.

KML2OFDS expects a KML file that contains both fibre optic routes and points of presence.

In broad strokes the script:

* parses a KML document for features and separates them into nodes (any point feature) and spans (any LineString or collection of LineStrings);
* checks for duplicate nodes based on node `name` and latitude/longitude with adjustable precision;
* snaps nodes to the closest point on the closest span if they are not already on a span;
* breaks each span at every point where a node intersects it, producing shorter spans;
* associates each span with start and end nodes;
* auto-generates nodes where a span lacks a start or end node (to comply with OFDS);
* adds metadata to spans and nodes;
* consolidates auto-generated nodes: merges those in close proximity, merges auto-generated nodes with nearby proper nodes, and treats auto-generated nodes near spans as forks.

## Configuration (Network Profile)

Create an INI file (e.g. `my_network.ini`) with at least:

```ini
[DEFAULT]
kml_file_name = your_network.kml
network_name = Your Network Name
network_id = your-network-id
output_name_prefix = YOUR-PREFIX

[DIRECTORY]
input_directory = input/
output_directory = output/
```

Key options in `default.profile`:

| Option | Description |
|--------|-------------|
| `kml_file_name` | KML file to convert (required) |
| `network_name`, `network_id` | Network metadata |
| `network_status` | e.g. `Operational` |
| `ignore_placemarks` | Semicolon-separated placemark names to ignore |
| `threshold_meters` | Distance (m) for consolidating auto-generated nodes (default: 5000) |
| `merge_contiguous_spans` | Merge spans that share endpoints |
| `rename_spans_from_nodes` | Name spans from start/end node names |
| `debug_enabled` | Write debug files to `debug_output_directory` |
| `validate_output` | Validate output against OFDS schema |

## Output

The pipeline produces three files in the output directory (with a date suffix):

* `{prefix}_ofds-nodes_{date}.geojson` — nodes (points of presence)
* `{prefix}_ofds-spans_{date}.geojson` — fibre spans
* `{prefix}_ofds-json_{date}.json` — OFDS JSON bundle

## Install kml2ofds

### Using pipx

The recommended way to install `kml2ofds` as a user is with [pipx][pipx].

First ensure [pipx is installed](https://pipx.pypa.io/stable/#install-pipx), then:

```sh
pipx install git+https://github.com/Open-Telecoms-Data/kml2ofds.git@main
```

Then run (you may need to restart your shell):

```sh
kml2ofds --network-profile default.profile
```

### Using pip

To install inside an existing Python virtual environment or conda environment:

```sh
# First activate your environment, then:
pip install git+https://github.com/Open-Telecoms-Data/kml2ofds.git@main
```

## Developing kml2ofds

Install the package as editable:

```sh
cd path/to/kml2ofds/
source .venv/bin/activate  # If using a Python virtual environment

pip install -e .
```

To add, remove, or update dependencies, edit the `dependencies` section in `pyproject.toml`, then run `pip install -e .` again.

### Running tests

```sh
pip install -e ".[test]"
pytest
```

[ofds-repo]: <https://github.com/Open-Telecoms-Data/open-fibre-data-standard>
[ofds-docs]: <https://open-fibre-data-standard.readthedocs.io/en/latest/reference/schema.html>
[pipx]: <https://github.com/pypa/pipx/>
