# TDL-020: NetworkX Graph Model

---
id: TDL-020
status: pending
priority: critical
phase: 2
depends_on: [TDL-012, TDL-013]
blocks: [TDL-021, TDL-022, TDL-023, TDL-024]
---

## Problem Statement

TouchDesigner networks are directed graphs. To validate structural properties (cycles, types, references), we need a graph representation. NetworkX is the standard Python graph library. This ticket defines the data model and graph construction algorithm.

## Acceptance Criteria

- [ ] `OperatorNode` dataclass captures operator properties
- [ ] `Connection` dataclass captures edges with input indices
- [ ] `build_network_graph()` constructs DiGraph from parsed .n files
- [ ] Node attributes include family, type, path, tile position
- [ ] Edge attributes include input index
- [ ] Missing references create placeholder nodes (for TDL-023)
- [ ] Graph builds successfully for entire sample corpus

## Files to Create

```
td_linter/
├── graph/
│   ├── __init__.py
│   ├── model.py           # OperatorNode, Connection, OperatorFamily
│   └── builder.py         # build_network_graph()
└── tests/
    └── test_graph_builder.py
```

## Research Pointers

### NetworkX Fundamentals

- https://networkx.org/documentation/stable/reference/classes/digraph.html
- `DiGraph` for directed graphs
- `G.add_node(id, **attributes)` - Add node with data
- `G.add_edge(u, v, **attributes)` - Add edge with data

### Data Model Design

Study the spec's model and consider:

**OperatorNode**: What properties do validators need?
- name (e.g., "displace1")
- family (TOP, CHOP, etc.)
- op_type (e.g., "displace")
- path (full path in network)
- tile (x, y, w, h for overlap detection)

**Connection**: What properties?
- source operator path
- target operator path
- input_index (which input slot)

### Graph Construction Algorithm

High-level approach:
1. **Phase 1**: Walk all .n files, create nodes
2. **Phase 2**: Walk all .n files again, create edges from `inputs` blocks

Why two phases? Edges reference nodes that must exist. Forward references would fail in single pass.

### Handling Missing References

When an input references an operator that doesn't exist:
- Create a placeholder node with `MISSING:` prefix
- Flag the edge as `missing=True`
- This allows TDL-023 to report dangling inputs without crashing

### Path Resolution

Inputs in .n files use relative names:
```
inputs
{
0   moviefilein1
1   chopto1
}
```

These must be resolved to full paths based on the .n file's location.

### Node ID Strategy

Options for node IDs:
1. Full path: `/project1/container1/displace1`
2. Relative path: `project1/container1/displace1`

Consider: What makes graph queries easiest?

### Graph Attributes

Use `G.graph` for metadata:
- `G.graph['toe_dir_path']` - Source directory
- `G.graph['td_version']` - If detectable

## Testing Strategy

| Test | Description |
|------|-------------|
| Empty project | No operators |
| Flat project | All operators at root |
| Nested project | Containers with children |
| Missing refs | References to non-existent ops |

## Performance Considerations

- Sample corpus may have projects with 1000+ operators
- NetworkX handles this fine, but test with large samples
- Consider lazy loading for very large projects

## Definition of Done

All acceptance criteria checked. Graph builds for all samples without exception.
