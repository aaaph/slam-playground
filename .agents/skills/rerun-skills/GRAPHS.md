# Graph Visualization

Detailed guide for visualizing graph structures with Rerun, including force-based layouts.

## GraphNodes - Define Graph Nodes

Define nodes in a graph with positions, colors, labels, and sizes.

### Basic Nodes

```python
import rerun as rr

# Simple node definition
nodes = ["A", "B", "C", "D"]
rr.log("graph", rr.GraphNodes(nodes))
```

### With Labels

```python
nodes = ["node_0", "node_1", "node_2"]
labels = ["Start", "Middle", "End"]

rr.log("graph", rr.GraphNodes(nodes, labels=labels))
```

### With Colors and Radii

```python
nodes = ["root", "child1", "child2", "child3"]
colors = [
    [255, 0, 0],  # Red for root
    [0, 255, 0],  # Green
    [0, 255, 0],  # Green
    [0, 0, 255],  # Blue
]
radii = [50, 30, 30, 40]

rr.log("tree", rr.GraphNodes(nodes, labels=nodes, colors=colors, radii=radii))
```

### With Fixed Positions

```python
# Manually position nodes (2D coordinates)
nodes = ["sunny", "rainy", "cloudy"]
positions = [[0, 0], [150, 150], [300, 0]]

rr.log("markov_chain", rr.GraphNodes(nodes, labels=nodes, positions=positions))
```

### Expected Data

- `node_ids`: List of strings (node identifiers)
- `labels`: List of strings (optional display names)
- `positions`: List of `[x, y]` or `(N, 2)` array (optional, for fixed layouts)
- `colors`: List of `[R, G, B]` or `(N, 3)` array
- `radii`: List of floats or single float

## GraphEdges - Define Graph Edges

Define edges connecting nodes.

### Basic Edges

```python
# Edges as (source, target) tuples
edges = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")]

rr.log("graph", rr.GraphEdges(edges, graph_type="undirected"))
```

### Directed Graphs

```python
edges = [
    ("start", "node1"),
    ("node1", "node2"),
    ("node2", "end"),
    ("node1", "end"),  # Skip connection
]

rr.log("dag", rr.GraphEdges(edges, graph_type="directed"))
```

### Full Graph Example

```python
# Combine nodes and edges
nodes = ["A", "B", "C", "D"]
edges = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")]

rr.log("graph", rr.GraphNodes(nodes, labels=nodes), rr.GraphEdges(edges, graph_type="undirected"))
```

## Force-Based Layouts

Rerun can automatically compute node positions using force-based layout algorithms.

### Automatic Layout

When you don't provide positions, Rerun computes them:

```python
# No positions specified - Rerun will compute layout
nodes = ["node_" + str(i) for i in range(20)]
edges = generate_random_edges(nodes)

rr.log("auto_graph", rr.GraphNodes(nodes), rr.GraphEdges(edges, graph_type="directed"))
```

### Configuring Forces via Blueprint

```python
import rerun.blueprint as rrb
from rerun.blueprint.archetypes import ForceLink, ForceManyBody, ForceCollisionRadius

# Configure force simulation
blueprint = rrb.Blueprint(
    rrb.GraphView(
        origin="graph",
        force_link=ForceLink(distance=60),  # Link distance
        force_many_body=ForceManyBody(strength=-60),  # Repulsion
        force_collision_radius=ForceCollisionRadius(enabled=True),  # Collision
    )
)

rr.send_blueprint(blueprint)
```

### Disabling Forces

For fixed layouts (e.g., Markov chains with known positions):

```python
blueprint = rrb.Blueprint(
    rrb.GraphView(
        origin="markov_chain", force_link=ForceLink(enabled=False), force_many_body=ForceManyBody(enabled=False)
    )
)
```

## Complete Examples

### Lattice Graph

```python
import itertools


# Create grid lattice
def log_lattice(num_nodes: int) -> None:
    coordinates = itertools.product(range(num_nodes), range(num_nodes))

    nodes, colors = zip(
        *[
            (
                str(i),
                rr.components.Color(
                    [round((x / (num_nodes - 1)) * 255), round((y / (num_nodes - 1)) * 255), 0, 255]
                ),
            )
            for i, (x, y) in enumerate(coordinates)
        ],
        strict=False,
    )

    rr.log(
        "lattice",
        rr.GraphNodes(
            nodes,
            colors=colors,
            labels=[f"({x}, {y})" for x, y in itertools.product(range(num_nodes), range(num_nodes))],
        ),
        static=True,
    )

    # Create edges connecting neighbors
    edges = []
    for x, y in itertools.product(range(num_nodes), range(num_nodes)):
        if y > 0:
            source = (y - 1) * num_nodes + x
            target = y * num_nodes + x
            edges.append((str(source), str(target)))
        if x > 0:
            source = y * num_nodes + (x - 1)
            target = y * num_nodes + x
            edges.append((str(source), str(target)))

    rr.log("lattice", rr.GraphEdges(edges, graph_type="directed"), static=True)


log_lattice(10)
```

### Growing Tree

```python
import random

# Randomly growing tree
nodes = ["root"]
radii = [42]
colors = [[81, 81, 81, 255]]
edges = []

for i in range(50):
    existing = random.choice(nodes)
    new_node = str(i)
    nodes.append(new_node)
    radii.append(random.randint(10, 50))
    colors.append([255, 127, 0, 255])
    edges.append((existing, new_node))

    rr.set_time("frame", sequence=i)
    rr.log(
        "tree",
        rr.GraphNodes(nodes, labels=nodes, radii=radii, colors=colors),
        rr.GraphEdges(edges, graph_type=rr.GraphType.Directed),
    )
```

### Markov Chain

```python
import numpy as np

# Markov chain with transition probabilities
transition_matrix = np.array(
    [
        [0.8, 0.1, 0.1],  # From sunny
        [0.3, 0.4, 0.3],  # From rainy
        [0.2, 0.3, 0.5],  # From cloudy
    ]
)

state_names = ["sunny", "rainy", "cloudy"]
positions = [[0, 0], [150, 150], [300, 0]]

inactive_color = [153, 153, 153, 255]
active_colors = [
    [255, 127, 0, 255],  # Orange
    [55, 126, 184, 255],  # Blue
    [152, 78, 163, 255],  # Purple
]

# Create edges for all transitions with prob > 0
edges = [
    (state_names[i], state_names[j])
    for i in range(len(state_names))
    for j in range(len(state_names))
    if transition_matrix[i][j] > 0
]
edges.append(("start", "sunny"))

# Simulate Markov chain
state = "sunny"
for i in range(50):
    current_idx = state_names.index(state)
    next_idx = np.random.choice(range(len(state_names)), p=transition_matrix[current_idx])
    state = state_names[next_idx]

    # Highlight current state
    colors = [inactive_color] * len(state_names)
    colors[next_idx] = active_colors[next_idx]

    rr.set_time("frame", sequence=i)
    rr.log(
        "markov_chain",
        rr.GraphNodes(state_names, labels=state_names, colors=colors, positions=positions),
        rr.GraphEdges(edges, graph_type="directed"),
    )
```

### Node-Link Diagram

```python
# Standard node-link diagram with automatic layout
nodes = ["A", "B", "C", "D", "E", "F"]
edges = [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"), ("C", "E"), ("C", "F"), ("E", "F")]

rr.log("node_link", rr.GraphNodes(nodes, labels=nodes), rr.GraphEdges(edges, graph_type="undirected"))

# Configure layout
blueprint = rrb.Blueprint(
    rrb.GraphView(
        origin="node_link",
        name="Node-Link Diagram",
        force_link=ForceLink(distance=60),
        force_many_body=ForceManyBody(strength=-60),
    )
)
```

### Bubble Chart (No Edges)

```python
# Bubble chart - nodes only, no edges
nodes = [f"item_{i}" for i in range(20)]
radii = [random.randint(10, 50) for _ in range(20)]
colors = [random.choice(color_palette) for _ in range(20)]

rr.log("bubble_chart", rr.GraphNodes(nodes, labels=nodes, radii=radii, colors=colors))

# Disable forces, use collision only
blueprint = rrb.Blueprint(
    rrb.GraphView(
        origin="bubble_chart",
        name="Bubble Chart",
        force_link=ForceLink(enabled=False),
        force_many_body=ForceManyBody(enabled=False),
        force_collision_radius=ForceCollisionRadius(enabled=True),
        defaults=[rr.GraphNodes.from_fields(show_labels=False)],
    )
)
```

## Dynamic Graphs

### Updating Over Time

```python
# Start with small graph
nodes = ["root"]
edges = []

for i in range(100):
    rr.set_time("frame", sequence=i)

    # Add new node every 10 frames
    if i % 10 == 0 and i > 0:
        new_node = f"node_{i}"
        nodes.append(new_node)
        # Connect to random existing node
        parent = random.choice(nodes[:-1])
        edges.append((parent, new_node))

    # Log current state
    rr.log("growing_graph", rr.GraphNodes(nodes, labels=nodes), rr.GraphEdges(edges, graph_type="directed"))
```

### Highlighting Nodes

```python
# Highlight nodes based on algorithm state
for step in algorithm_steps:
    rr.set_time("step", sequence=step.index)

    # Default gray color
    colors = [[128, 128, 128]] * len(nodes)

    # Highlight active nodes
    for active_node in step.active_nodes:
        idx = nodes.index(active_node)
        colors[idx] = [255, 0, 0]  # Red

    # Highlight visited nodes
    for visited_node in step.visited_nodes:
        idx = nodes.index(visited_node)
        colors[idx] = [0, 255, 0]  # Green

    rr.log(
        "algorithm/graph",
        rr.GraphNodes(nodes, labels=nodes, colors=colors),
        rr.GraphEdges(edges, graph_type="directed"),
    )
```

## Multiple Graph Views

Show the same graph with different layouts:

```python
import rerun.blueprint as rrb
from rerun.blueprint.archetypes import ForceLink, ForceManyBody, ForceCollisionRadius

blueprint = rrb.Blueprint(
    rrb.Grid(
        # Force-directed layout
        rrb.GraphView(
            origin="graph",
            name="Force Layout",
            force_link=ForceLink(distance=60),
            force_many_body=ForceManyBody(strength=-60),
        ),
        # Circular layout (disable forces, use fixed positions)
        rrb.GraphView(
            origin="graph_circular",
            name="Circular Layout",
            force_link=ForceLink(enabled=False),
            force_many_body=ForceManyBody(enabled=False),
        ),
        # Hierarchical (tree-like)
        rrb.GraphView(
            origin="graph_tree",
            name="Tree Layout",
            force_link=ForceLink(distance=100),
            force_many_body=ForceManyBody(strength=-200),
        ),
    )
)
```

## Tips and Best Practices

### Node IDs vs Labels

- `node_ids`: Unique identifiers (use for edges)
- `labels`: Display names (can be the same or more descriptive)

```python
nodes = ["user_123", "user_456", "user_789"]
labels = ["Alice", "Bob", "Charlie"]

rr.log("social_graph", rr.GraphNodes(nodes, labels=labels))
```

### Choosing Graph Type

- `"directed"`: Arrows show direction (DAGs, flow graphs)
- `"undirected"`: No direction (social networks, meshes)

### Performance

- For large graphs (>1000 nodes), consider:
  - Using `static=True` if the graph doesn't change
  - Simplifying the graph (reduce edges)
  - Disabling labels for cleaner view

### Color Schemes

```python
# Categorical colors for different types
color_scheme = [
    [228, 26, 28, 255],  # Red
    [55, 126, 184, 255],  # Blue
    [77, 175, 74, 255],  # Green
    [152, 78, 163, 255],  # Purple
    [255, 127, 0, 255],  # Orange
]

# Assign colors by node type
colors = [color_scheme[node_type] for node_type in node_types]
```
