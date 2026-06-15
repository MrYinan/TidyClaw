import json
import shutil
import unittest
from pathlib import Path

from scripts.inspection_waypoints import (
    INSPECTION_WAYPOINTS_SCHEMA,
    WAYPOINT_SOURCE_AUTHORED,
    WAYPOINT_SOURCE_GENERATED,
    build_inspection_waypoints,
    graph_from_snapshot,
    waypoint_id_for_cell,
)
from scripts.map_backend import MapSnapshot
from scripts.position_map_core import CELL_FREE, edge_key, four_neighbors


def grid_snapshot(width: int = 5, height: int = 5) -> MapSnapshot:
    cells = {}
    for x in range(width):
        for z in range(height):
            cell = f"{x},{z}"
            cells[cell] = {
                "cell": cell,
                "state": CELL_FREE,
                "visited": cell == "0,0",
                "world_position": {"x": x * 0.25, "y": 0.9, "z": z * 0.25},
            }
    open_edges = []
    for cell in cells:
        for neighbor in four_neighbors(cell):
            if neighbor in cells:
                open_edges.append(edge_key(cell, neighbor))
    position_status = {
        "pose": {"cell": "0,0", "heading": "north"},
        "cells": cells,
        "edges": {"known_open_edges": open_edges, "blocked_edges": []},
        "frontiers": [],
        "stats": {"free_cell_count": len(cells), "visited_cell_count": 1},
    }
    return MapSnapshot(
        backend="map_bundle",
        generated_at="2026-06-13T00:00:00+08:00",
        map_frame={"coordinate_mode": "grid_xz", "cell_size_m": 0.25},
        pose={"cell": "0,0", "heading": "north"},
        coverage={"free_cell_count": len(cells), "visited_cell_count": 1},
        frontiers=[],
        edges={"known_open_edges": open_edges, "blocked_edges": []},
        position_status=position_status,
        room_state={"last_cell": "0,0", "last_heading": "north"},
    )


class InspectionWaypointsTests(unittest.TestCase):
    def test_generates_stable_coverage_waypoints_from_reachable_cells(self) -> None:
        snapshot = grid_snapshot()

        first = build_inspection_waypoints(snapshot, radius_cells=2, max_waypoints=12)
        second = build_inspection_waypoints(snapshot, radius_cells=2, max_waypoints=12)

        self.assertEqual(first["schema"], INSPECTION_WAYPOINTS_SCHEMA)
        self.assertEqual(first["source"], WAYPOINT_SOURCE_GENERATED)
        self.assertEqual(first["free_cell_count"], 25)
        self.assertGreater(first["required_waypoint_count"], 1)
        self.assertLess(first["required_waypoint_count"], 25)
        self.assertEqual(first["required_waypoint_ids"], second["required_waypoint_ids"])
        for waypoint in first["waypoints"]:
            self.assertIn("waypoint_id", waypoint)
            self.assertIn("cell", waypoint)
            self.assertEqual(waypoint["purpose"], "coverage_scan")
            self.assertEqual(waypoint["waypoint_source"], WAYPOINT_SOURCE_GENERATED)
            self.assertGreater(waypoint["covered_cell_count"], 0)
            self.assertGreater(waypoint["coverage_estimate"], 0.0)

    def test_authored_waypoints_are_preferred_when_present(self) -> None:
        snapshot = grid_snapshot()
        snapshot = MapSnapshot(
            backend=snapshot.backend,
            generated_at=snapshot.generated_at,
            map_frame=snapshot.map_frame,
            pose=snapshot.pose,
            coverage=snapshot.coverage,
            frontiers=snapshot.frontiers,
            edges=snapshot.edges,
            position_status=snapshot.position_status,
            room_state={
                **snapshot.room_state,
                "inspection_waypoints": [
                    {
                        "waypoint_id": "public_wp_counter",
                        "cell": "2,2",
                        "label": "counter inspection",
                        "purpose": "fixture_coverage",
                        "coverage_estimate": 0.25,
                    }
                ],
            },
        )

        result = build_inspection_waypoints(snapshot, radius_cells=2, max_waypoints=12)

        self.assertEqual(result["source"], WAYPOINT_SOURCE_AUTHORED)
        self.assertEqual(result["required_waypoint_ids"], ["public_wp_counter"])
        self.assertEqual(result["waypoints"][0]["cell"], "2,2")
        self.assertEqual(result["waypoints"][0]["purpose"], "fixture_coverage")

    def test_graph_honors_known_open_edges(self) -> None:
        snapshot = grid_snapshot(width=2, height=1)
        graph = graph_from_snapshot(snapshot)

        self.assertIn("0,0", graph.adjacency)
        self.assertIn("1,0", graph.adjacency["0,0"])
        self.assertEqual(graph.free_cell_count, 2)

    def test_waypoint_id_for_cell_encodes_negative_cells(self) -> None:
        self.assertEqual(waypoint_id_for_cell("-2,5"), "wp_xm2_z5")
        self.assertEqual(waypoint_id_for_cell("2,-5"), "wp_x2_zm5")


if __name__ == "__main__":
    unittest.main()
