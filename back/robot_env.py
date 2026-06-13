from __future__ import annotations

import base64
import math
import os
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ai2thor.controller import Controller
import numpy as np
from PIL import Image


JsonDict = Dict[str, Any]
EXPLICIT_SURFACE_PLACE_SOURCES = {
    "depth_region_geometry",
    "pointcloud_plane",
    "pointcloud_plane_completion",
    "pointcloud_plane_grid_completion",
}


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_list(name: str, default: Sequence[str]) -> Set[str]:
    value = os.getenv(name)
    if not value:
        return set(default)
    return {item.strip() for item in value.split(",") if item.strip()}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


class RobotEnvironment:
    """
    AI2-THOR backend environment wrapper for the OpenClaw robot-cleaner V2.

    Design boundary:
    - This class is allowed to use AI2-THOR metadata internally because it is the
      simulator-side actuator/evaluator layer.
    - Online Agent/Skill code must use /observation and sanitized /move,/clean
      responses, never raw metadata.
    - /eval/state and /scenario/* are offline evaluation/debug/benchmark control
      interfaces.

    V2 scenario goal:
    The backend no longer represents a single fixed demo. It supports a
    single-room multi-condition benchmark: empty room, front/side/far floor
    trash, non-floor decoys, and obstacle/collision-recovery situations. This is
    sufficient for the V2 experiments while avoiding premature multi-room scope.
    """

    DEFAULT_CLEANABLE_PROXY_TYPES = (
        "Tomato",
        "Apple"
    )
    DEFAULT_NON_FLOOR_PROXY_TYPES = (
        "Apple",
        "Tomato",
        "Potato",
        "Book",
        "Mug",
        "Bowl",
        "Cup",
    )
    DEFAULT_OBSTACLE_TYPES = (
        "Chair",
        "Table",
        "Sofa",
        "ArmChair",
        "Cabinet",
        "Drawer",
        "Dishwasher",
        "Microwave",
        "Stove",
        "Shelf",
        "ShelvingUnit",
        "CounterTop",
        "DiningTable",
    )
    DEFAULT_PICKUP_TARGET_TYPES = (
        "Book",
        "Apple",
        "Tomato",
        "Potato",
        "Mug",
        "Cup",
        "Plate",
    )
    DEFAULT_PLACE_RECEPTACLE_TYPES = (
        "DiningTable",
        "CoffeeTable",
        "SideTable",
        "CounterTop",
        "Sink",
        "Bowl",
        "Plate",
    )
    DEFAULT_SERVICE_ANNOTATION_TYPES = (
        "Book",
        "Apple",
        "Tomato",
        "Potato",
        "Mug",
        "Cup",
        "Bowl",
        "Plate",
        "DiningTable",
        "CoffeeTable",
        "SideTable",
        "CounterTop",
        "Sink",
        "SinkBasin",
        "Chair",
        "ArmChair",
        "Sofa",
        "Cabinet",
        "Drawer",
        "Dishwasher",
        "Microwave",
        "Stove",
        "StoveBurner",
        "StoveKnob",
        "Shelf",
        "ShelvingUnit",
        "Bottle",
        "SoapBottle",
        "Vase",
        "HousePlant",
        "Kettle",
        "Lettuce",
        "Pan",
        "Pot",
        "ButterKnife",
        "Spatula",
        "PepperShaker",
        "SaltShaker",
        "PaperTowelRoll",
        "Bed",
        "Table",
    )

    # Keep the historical name used by earlier V1/V2 code.
    TRASH_OBJECT_TYPES = set(DEFAULT_CLEANABLE_PROXY_TYPES)

    def __init__(
        self,
        scene: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        mode: Optional[str] = None,
        seed_trash: Optional[bool] = None,
    ) -> None:
        self.scene = scene or os.getenv("ROBOT_SCENE", "FloorPlan1")
        self.width = int(width or os.getenv("ROBOT_VIEW_WIDTH", "600"))
        self.height = int(height or os.getenv("ROBOT_VIEW_HEIGHT", "600"))
        self.mode = (mode or os.getenv("ROBOT_ENV_MODE", "debug")).strip().lower()
        self.cleanable_object_types = _env_list("ROBOT_CLEANABLE_TYPES", self.DEFAULT_CLEANABLE_PROXY_TYPES)
        self.non_floor_proxy_types = _env_list("ROBOT_NON_FLOOR_PROXY_TYPES", self.DEFAULT_NON_FLOOR_PROXY_TYPES)
        self.obstacle_object_types = _env_list("ROBOT_OBSTACLE_TYPES", self.DEFAULT_OBSTACLE_TYPES)
        self.pickup_target_types = _env_list("ROBOT_PICKUP_TARGET_TYPES", self.DEFAULT_PICKUP_TARGET_TYPES)
        self.place_receptacle_types = _env_list("ROBOT_PLACE_RECEPTACLE_TYPES", self.DEFAULT_PLACE_RECEPTACLE_TYPES)
        self.service_annotation_types = _env_list(
            "ROBOT_SERVICE_ANNOTATION_TYPES",
            self.DEFAULT_SERVICE_ANNOTATION_TYPES,
        )
        self.TRASH_OBJECT_TYPES = set(self.cleanable_object_types)

        if seed_trash is None:
            seed_trash = _env_bool("ROBOT_SEED_TRASH", self.mode == "debug")
        self.seed_trash = bool(seed_trash)

        self.current_scenario: JsonDict = {
            "name": os.getenv("ROBOT_SCENARIO", "manual_debug" if self.mode == "debug" else "manual_formal"),
            "source": "env_or_default",
        }
        self.seeded_objects: List[JsonDict] = []
        self._used_seed_object_ids: Set[str] = set()

        print(
            f"正在初始化 AI2-THOR 环境: {self.scene}, mode={self.mode}, seed_trash={self.seed_trash}",
            flush=True,
        )
        self.controller = Controller(
            agentMode="default",
            visibilityDistance=float(os.getenv("ROBOT_VISIBILITY_DISTANCE", "1.5")),
            scene=self.scene,
            gridSize=float(os.getenv("ROBOT_GRID_SIZE", "0.25")),
            width=self.width,
            height=self.height,
            renderInstanceSegmentation=True,
            renderObjectImage=True,
            renderClassImage=True,
            renderDepthImage=True,
        )
        self._initialize_rendering()
        self.last_event = self.controller.step(action="Pass")

        if _env_bool("ROBOT_REMOVE_ORIGINAL_OBJECTS_ON_LOAD", True):
            self.remove_original_objects_on_load()

        if self.mode == "debug":
            self.teleport_to_debug_pose()

        if self.seed_trash:
            self.spawn_task_object(kind="floor_trash", layout="front-center", distance=0.6)

        if _env_bool("ROBOT_LOOK_DOWN_ON_START", True):
            self.look_down_once()
        self.last_event = self.controller.step(action="Pass")
        print("环境初始化完成，机器人已就位。", flush=True)

    # ------------------------------------------------------------------
    # Basic perception
    # ------------------------------------------------------------------

    def _rendering_kwargs(self) -> JsonDict:
        return {
            "gridSize": float(os.getenv("ROBOT_GRID_SIZE", "0.25")),
            "visibilityDistance": float(os.getenv("ROBOT_VISIBILITY_DISTANCE", "1.5")),
            "renderInstanceSegmentation": True,
            "renderObjectImage": True,
            "renderClassImage": True,
            "renderDepthImage": True,
        }

    def _initialize_rendering(self) -> None:
        """Make segmentation/bbox rendering explicit after construction/reset."""
        try:
            self.last_event = self.controller.step(action="Initialize", **self._rendering_kwargs())
        except Exception as exc:
            print(f"[rendering] Initialize skipped: {exc}", flush=True)

    def get_first_person_view(self) -> Image.Image:
        frame = self.last_event.frame
        return Image.fromarray(frame)

    def get_first_person_view_base64(self) -> str:
        img = self.get_first_person_view()
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def get_depth_frame(self) -> Optional[np.ndarray]:
        depth = getattr(self.last_event, "depth_frame", None)
        if depth is None:
            return None
        try:
            frame = np.asarray(depth, dtype=np.float32)
        except Exception:
            return None
        if frame.ndim != 2 or frame.size == 0:
            return None
        return frame

    def get_depth_frame_base64_npy(self) -> Optional[str]:
        """Return the current depth frame as base64-encoded .npy bytes.

        AI2-THOR depth values are metric distances in meters. Keeping the data
        as float32 .npy avoids losing precision in the online geometry layer.
        """
        frame = self.get_depth_frame()
        if frame is None:
            return None
        buffered = BytesIO()
        np.save(buffered, frame.astype(np.float32, copy=False))
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def get_robot_state(self) -> JsonDict:
        metadata = self.last_event.metadata
        agent_info = metadata["agent"]
        return {
            "position": agent_info.get("position", {}),
            "rotation": agent_info.get("rotation", {}),
            "cameraHorizon": agent_info.get("cameraHorizon"),
        }

    def get_state_snapshot(self) -> JsonDict:
        objects = self.last_event.metadata.get("objects", [])
        visible_cleanable = [
            self._summarize_candidate_object(obj)
            for obj in objects
            if obj.get("visible", False) and obj.get("objectType") in self.cleanable_object_types
        ]
        visible_obstacles = [
            self._summarize_object(obj)
            for obj in objects
            if obj.get("visible", False) and obj.get("objectType") in self.obstacle_object_types
        ]
        visible_pickup_targets = [
            self._summarize_service_object(obj, task_semantic_class="pickup_target")
            for obj in objects
            if obj.get("visible", False) and obj.get("objectType") in self.pickup_target_types
        ]
        visible_receptacles = [
            self._summarize_service_object(obj, task_semantic_class="place_receptacle")
            for obj in objects
            if obj.get("visible", False)
            and (
                obj.get("objectType") in self.place_receptacle_types
                or bool(obj.get("receptacle", False))
            )
        ]
        annotation_objects = [
            self._summarize_annotation_object(obj)
            for obj in objects
            if str(obj.get("objectType") or "") in self.service_annotation_types
        ]
        return {
            "status": "success",
            "schema_version": 2,
            "result_type": "eval_state_snapshot",
            "scene": self.scene,
            "mode": self.mode,
            "scenario": self.current_scenario,
            "robot": self.get_robot_state(),
            "task_class_schema": {
                "pickup_target": "visible household object that can be picked up or tidied",
                "place_receptacle": "visible support/container candidate for placement",
                "obstacle": "large object or geometry that constrains movement",
                "cleanable_object": "legacy cleanable floor proxy used by the clean skill",
                "ignored_object": "visible object that is not relevant to the current task",
            },
            "service_task_alignment": {
                "main_dataset": "AI2-THOR generated RGB + offline metadata labels",
                "online_policy": "Agent decisions use RGB perception and sanitized action feedback.",
                "legacy_cleaning": "The clean skill remains available as a service action.",
            },
            "cleanable_object_types": sorted(self.cleanable_object_types),
            "non_floor_proxy_types": sorted(self.non_floor_proxy_types),
            "obstacle_object_types": sorted(self.obstacle_object_types),
            "pickup_target_types": sorted(self.pickup_target_types),
            "place_receptacle_types": sorted(self.place_receptacle_types),
            "service_annotation_types": sorted(self.service_annotation_types),
            "frame": {
                "width": int(self.width),
                "height": int(self.height),
            },
            "segmentation_debug": self._segmentation_debug(),
            "annotation_objects": annotation_objects,
            "instance_detections2D": self._instance_detections_2d(),
            "visible_trash_candidates": visible_cleanable,
            "visible_obstacle_candidates": visible_obstacles[:20],
            "visible_pickup_candidates": visible_pickup_targets[:20],
            "visible_receptacle_candidates": visible_receptacles[:20],
            "inventory": self.get_inventory_state(),
            "seeded_objects": list(self.seeded_objects),
            "available_seed_proxy_types": self.available_object_type_counts(
                sorted(
                    self.cleanable_object_types
                    | self.non_floor_proxy_types
                    | self.pickup_target_types
                    | self.place_receptacle_types
                )
            ),
        }

    def get_groundtruth_map_snapshot(self) -> JsonDict:
        """Return an offline-only AI2-THOR navigation map projection.

        This endpoint is intentionally separate from online observation.  It is
        meant for an offline mapping stage or controlled simulator ablation:
        AI2-THOR reachable positions define the known traversable floor cells,
        while online tidy decisions still consume the normalized MapBackend
        contract instead of raw simulator metadata.
        """

        try:
            event = self.controller.step(action="GetReachablePositions")
            metadata = event.metadata if hasattr(event, "metadata") and isinstance(event.metadata, dict) else {}
            reachable = metadata.get("actionReturn") or []
            success = bool(metadata.get("lastActionSuccess", False))
            error_message = str(metadata.get("errorMessage") or "")
        except Exception as exc:
            return {
                "status": "error",
                "schema_version": 1,
                "result_type": "error_ai2thor_groundtruth_map_failed",
                "message": str(exc),
                "online_safe": False,
                "usage_scope": "offline_mapping_only",
                "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            }

        positions: List[JsonDict] = []
        if isinstance(reachable, list):
            for item in reachable:
                if not isinstance(item, dict):
                    continue
                try:
                    positions.append(
                        {
                            "x": round(float(item.get("x", 0.0)), 4),
                            "y": round(float(item.get("y", 0.0)), 4),
                            "z": round(float(item.get("z", 0.0)), 4),
                        }
                    )
                except (TypeError, ValueError):
                    continue

        return {
            "status": "success" if success else "error",
            "schema_version": 1,
            "result_type": "ai2thor_groundtruth_map",
            "scene": self.scene,
            "mode": self.mode,
            "scenario": self.current_scenario,
            "robot": self.get_robot_state(),
            "grid_size_m": float(os.getenv("ROBOT_GRID_SIZE", "0.25")),
            "reachable_positions": positions,
            "reachable_position_count": len(positions),
            "lastActionSuccess": success,
            "error_message": error_message,
            "online_safe": False,
            "usage_scope": "offline_mapping_only",
            "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "warning": "Do not feed raw AI2-THOR metadata into online Agent decisions; use MapBackend/MapBundle output.",
        }

    # ------------------------------------------------------------------
    # Scenario/benchmark control
    # ------------------------------------------------------------------

    def reset_env(
        self,
        *,
        scene: Optional[str] = None,
        mode: Optional[str] = None,
        seed_trash: Optional[bool] = None,
        initial_pose: Optional[JsonDict] = None,
        look_down: bool = True,
    ) -> JsonDict:
        if scene:
            self.scene = str(scene)
        if mode:
            self.mode = str(mode).strip().lower()
        if seed_trash is not None:
            self.seed_trash = bool(seed_trash)

        self.seeded_objects = []
        self._used_seed_object_ids = set()
        self.controller.reset(scene=self.scene)
        self._initialize_rendering()
        self.last_event = self.controller.step(action="Pass")

        if _env_bool("ROBOT_REMOVE_ORIGINAL_OBJECTS_ON_LOAD", True):
            self.remove_original_objects_on_load()

        if initial_pose:
            self.teleport_agent(initial_pose)
        elif self.mode == "debug":
            self.teleport_to_debug_pose()

        if self.seed_trash:
            self.spawn_task_object(kind="floor_trash", layout="front-center", distance=0.6)

        if look_down:
            self.look_down_once()
        self.last_event = self.controller.step(action="Pass")
        return {
            "status": "success",
            "schema_version": 2,
            "result_type": "env_reset",
            "scene": self.scene,
            "mode": self.mode,
            "seed_trash": self.seed_trash,
            "robot": self.get_robot_state(),
        }

    def apply_scenario(self, scenario: JsonDict) -> JsonDict:
        """Reset the simulator and apply a V2 benchmark scenario config."""
        if not isinstance(scenario, dict):
            return {"status": "error", "result_type": "error_invalid_scenario", "message": "scenario must be a JSON object"}

        name = str(scenario.get("name") or "unnamed_scenario")
        scene = str(scenario.get("scene") or self.scene)
        mode = str(scenario.get("mode") or "benchmark")
        look_down = bool(scenario.get("look_down", True))
        initial_pose = scenario.get("initial_pose") if isinstance(scenario.get("initial_pose"), dict) else None

        reset_result = self.reset_env(
            scene=scene,
            mode=mode,
            seed_trash=False,
            initial_pose=initial_pose,
            look_down=look_down,
        )
        self.current_scenario = {
            "name": name,
            "description": scenario.get("description", ""),
            "scene": scene,
            "mode": mode,
            "source": scenario.get("source", "configs/scenarios_v2.json"),
        }

        seed_results: List[JsonDict] = []
        for seed_spec in scenario.get("seed_objects", []) or []:
            if isinstance(seed_spec, dict):
                seed_results.append(self.spawn_task_object(**seed_spec))

        pre_actions = scenario.get("pre_actions", []) or []
        pre_action_results: List[JsonDict] = []
        for action in pre_actions:
            if isinstance(action, str):
                pre_action_results.append(self.execute_action(action))

        self.last_event = self.controller.step(action="Pass")
        return {
            "status": "success",
            "schema_version": 2,
            "result_type": "scenario_loaded",
            "scenario": self.current_scenario,
            "reset": reset_result,
            "seed_results": seed_results,
            "pre_action_results": pre_action_results,
            "robot": self.get_robot_state(),
        }

    def teleport_agent(self, pose: JsonDict) -> JsonDict:
        position = pose.get("position") or {}
        rotation = pose.get("rotation") or {}
        horizon = pose.get("cameraHorizon")
        kwargs: JsonDict = {
            "action": "Teleport",
            "position": {
                "x": float(position.get("x", -1.0)),
                "y": float(position.get("y", 0.9)),
                "z": float(position.get("z", 0.0)),
            },
            "rotation": {
                "x": float(rotation.get("x", 0.0)),
                "y": float(rotation.get("y", 180.0)),
                "z": float(rotation.get("z", 0.0)),
            },
        }
        if horizon is not None:
            kwargs["horizon"] = float(horizon)
        event = self.controller.step(**kwargs)
        self.last_event = event
        return {
            "status": "success" if event.metadata.get("lastActionSuccess", False) else "error",
            "result_type": "agent_teleport",
            "lastActionSuccess": event.metadata.get("lastActionSuccess", False),
            "error_message": event.metadata.get("errorMessage", ""),
            "robot": self.get_robot_state(),
        }

    def teleport_to_debug_pose(self) -> JsonDict:
        print("🤖 机器人正在瞬移到调试开阔区域...", flush=True)
        return self.teleport_agent(
            {
                "position": {"x": -1.0, "y": 0.9, "z": 0.0},
                "rotation": {"x": 0, "y": 180, "z": 0},
            }
        )

    def look_down_once(self) -> JsonDict:
        print("🤖 机器人正在低头观察地面...", flush=True)
        event = self.controller.step(action="LookDown")
        self.last_event = event
        return {
            "status": "success" if event.metadata.get("lastActionSuccess", False) else "error",
            "result_type": "look_down",
            "lastActionSuccess": event.metadata.get("lastActionSuccess", False),
            "error_message": event.metadata.get("errorMessage", ""),
            "robot": self.get_robot_state(),
        }

    def spawn_trash_in_front(self, distance: float = 0.6) -> JsonDict:
        return self.spawn_task_object(kind="floor_trash", layout="front-center", distance=distance)

    def remove_original_objects_on_load(self) -> JsonDict:
        """Hide unwanted original AI2-THOR scene objects after scene load/reset.

        Instead of RemoveFromScene, move them underground. This is usually more
        stable for startup cleanup in AI2-THOR.
        """
        remove_types = _env_list("ROBOT_REMOVE_ORIGINAL_OBJECT_TYPES", ("Bread", "CoffeeMachine", "Toaster"))

        removed: List[JsonDict] = []
        skipped: List[JsonDict] = []

        self.last_event = self.controller.step(action="Pass")
        objects = list(self.last_event.metadata.get("objects", []))

        for obj in objects:
            obj_type = str(obj.get("objectType") or "")
            obj_id = str(obj.get("objectId") or "")

            if obj_type not in remove_types:
                continue
            if not obj_id:
                continue

            print(f"[scene_cleanup] hiding original object: {obj_id}", flush=True)

            event = self.controller.step(
                action="TeleportObject",
                objectId=obj_id,
                position={"x": 0.0, "y": -10.0, "z": 0.0},
                rotation={"x": 0.0, "y": 0.0, "z": 0.0},
                forceAction=True,
            )

            record = {
                "objectId": obj_id,
                "objectType": obj_type,
                "lastActionSuccess": bool(event.metadata.get("lastActionSuccess", False)),
                "error_message": event.metadata.get("errorMessage", ""),
                "method": "teleport_underground",
            }

            if record["lastActionSuccess"]:
                removed.append(record)
            else:
                skipped.append(record)

        self.last_event = self.controller.step(action="Pass")

        result = {
            "status": "success",
            "schema_version": 2,
            "result_type": "original_scene_objects_hidden",
            "remove_types": sorted(remove_types),
            "removed_count": len(removed),
            "removed": removed,
            "skipped_count": len(skipped),
            "skipped": skipped,
        }
        print(f"[scene_cleanup] {result}", flush=True)
        return result

    def spawn_task_object(
        self,
        *,
        kind: str = "floor_trash",
        layout: str = "front-center",
        distance: float = 0.6,
        object_type: Optional[str] = None,
        object_types: Optional[Sequence[str]] = None,
        dataset_label: Optional[str] = None,
        task_semantic_class: Optional[str] = None,
        y: Optional[float] = None,
        lateral: Optional[float] = None,
    ) -> JsonDict:
        """Place an existing movable object into a controlled benchmark pose.

        kind values include the legacy cleaning proxies plus service-task
        labels such as pickup_target, place_receptacle and obstacle.

        Large receptacles/obstacles are often better supplied by the scene
        itself; TeleportObject is most reliable for pickup-sized objects.
        """
        kind = str(kind or "floor_trash")
        layout = str(layout or "front-center")
        distance = float(distance)

        if object_type:
            candidates = [str(object_type)]
        elif object_types:
            candidates = [str(item) for item in object_types]
        elif kind == "pickup_target":
            candidates = list(self.pickup_target_types)
        elif kind == "place_receptacle":
            candidates = list(self.place_receptacle_types)
        elif kind == "obstacle":
            candidates = list(self.obstacle_object_types)
        elif kind == "non_floor_decoy":
            candidates = list(self.non_floor_proxy_types)
        else:
            candidates = list(self.cleanable_object_types)

        target_obj = self._select_seed_object(candidates)
        if target_obj is None:
            msg = f"No available seed object found for types={candidates}."
            print(f"⚠️ {msg}", flush=True)
            return {
                "status": "error",
                "schema_version": 2,
                "result_type": "error_no_seed_object",
                "kind": kind,
                "layout": layout,
                "requested_object_types": candidates,
                "available_proxy_types": self.available_object_type_counts(candidates),
                "message": msg,
            }

        target_position = self._target_position_from_layout(layout=layout, distance=distance, y=y, lateral=lateral, kind=kind)
        event = self.controller.step(
            action="TeleportObject",
            objectId=target_obj["objectId"],
            position=target_position,
            rotation={"x": 0, "y": 0, "z": 0},
            forceAction=True,
        )
        self.last_event = self.controller.step(action="Pass")

        success = bool(event.metadata.get("lastActionSuccess", False))
        result = {
            "status": "success" if success else "error",
            "schema_version": 2,
            "result_type": "scenario_seed_object" if success else "error_seed_object_failed",
            "kind": kind,
            "layout": layout,
            "lastActionSuccess": success,
            "error_message": event.metadata.get("errorMessage", ""),
            "target_object_id": target_obj.get("objectId"),
            "target_object_type": target_obj.get("objectType"),
            "target_position": target_position,
            "dataset_label": str(dataset_label or target_obj.get("objectType") or kind),
            "task_semantic_class": str(task_semantic_class or self._task_class_for_seed_kind(kind)),
            "robot": self.get_robot_state(),
        }
        if success:
            self._used_seed_object_ids.add(str(target_obj.get("objectId")))
            self.seeded_objects.append(result)
        print(f"[scenario_seed_object] {result}", flush=True)
        return result

    def available_object_type_counts(self, object_types: Optional[Iterable[str]] = None) -> JsonDict:
        wanted = set(object_types or [])
        counts: Dict[str, int] = {}
        for obj in self.last_event.metadata.get("objects", []):
            obj_type = str(obj.get("objectType"))
            if wanted and obj_type not in wanted:
                continue
            counts[obj_type] = counts.get(obj_type, 0) + 1
        return dict(sorted(counts.items()))

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def execute_action(self, action_name: str, **kwargs: Any) -> JsonDict:
        valid_actions = {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight", "RotateLeft", "RotateRight", "LookUp", "LookDown"}
        if action_name not in valid_actions:
            return {
                "status": "error",
                "result_type": "error_invalid_action",
                "action": action_name,
                "message": f"Invalid action: {action_name}",
            }

        before = self.get_robot_state()
        print(f"[move] 执行动作: {action_name} 参数: {kwargs}", flush=True)
        event = self.controller.step(action=action_name, **kwargs)
        self.last_event = event
        after = self.get_robot_state()

        success = bool(event.metadata.get("lastActionSuccess", False))
        error_message = event.metadata.get("errorMessage", "")
        state_changed = self._robot_state_changed(before, after)

        if success:
            result_type = "move_executed"
            status = "success"
            message = f"已执行动作: {action_name}"
        else:
            result_type = "error_action_invalid_or_collision"
            status = "collision"
            message = error_message or "动作失败，前方可能有障碍物。"

        result = {
            "status": status,
            "result_type": result_type,
            "action": action_name,
            "lastActionSuccess": success,
            "message": message,
            "error_message": error_message,
            "position_before": before.get("position"),
            "rotation_before": before.get("rotation"),
            "position_after": after.get("position"),
            "rotation_after": after.get("rotation"),
            "state_changed": state_changed,
        }
        print(f"[move] result={result}", flush=True)
        return result

    # ------------------------------------------------------------------
    # Simulated cleaning actuator
    # ------------------------------------------------------------------

    def clean_trash_in_front(self) -> JsonDict:
        print("[clean] 开始执行 V2 仿真清扫执行器", flush=True)
        candidates = self._get_visible_trash_candidates()
        if not candidates:
            return {
                "status": "error",
                "result_type": "error_no_target_in_front",
                "message": "当前视野内没有发现可清扫垃圾代理物。",
                "candidates": [],
            }

        eligible = [c for c in candidates if c["clean_rule_passed"]]
        if not eligible:
            result_type = self._infer_clean_reject_type(candidates)
            return {
                "status": "error",
                "result_type": result_type,
                "message": self._clean_reject_message(result_type),
                "candidates": candidates,
            }

        eligible.sort(key=lambda c: c["distance"])
        target = eligible[0]
        target_obj_id = target["objectId"]
        print(f"[clean] 锁定合格目标: {target_obj_id}", flush=True)

        event = self.controller.step(
            action="TeleportObject",
            objectId=target_obj_id,
            position={"x": 0, "y": -10, "z": 0},
            rotation={"x": 0, "y": 0, "z": 0},
        )
        action_success = bool(event.metadata.get("lastActionSuccess", False))
        error_message = event.metadata.get("errorMessage", "")
        self.last_event = self.controller.step(action="Pass")

        target_after = self._find_object_by_id(target_obj_id)
        if target_after is None:
            return {
                "status": "success",
                "result_type": "clean_executed",
                "message": f"清扫成功，目标 {target_obj_id} 已从 metadata 中消失。",
                "target": target,
                "lastActionSuccess": action_success,
                "removed_from_view": True,
                "removed_from_scene": True,
                "error_message": error_message,
            }

        still_visible = bool(target_after.get("visible", False))
        pos_after = target_after.get("position", {})
        y_after = pos_after.get("y", None)
        removed_from_view = not still_visible
        moved_underground = y_after is not None and float(y_after) < -5
        clean_success = removed_from_view or moved_underground

        if clean_success:
            return {
                "status": "success",
                "result_type": "clean_executed",
                "message": f"清扫成功，目标 {target_obj_id} 已从当前可见区域移除。",
                "target": target,
                "lastActionSuccess": action_success,
                "removed_from_view": removed_from_view,
                "removed_from_scene": False,
                "target_after": self._summarize_object(target_after),
                "error_message": error_message,
            }

        if not action_success:
            return {
                "status": "error",
                "result_type": "error_ai2thor_clean_failed",
                "message": f"AI2-THOR 清扫动作失败：{error_message}",
                "target": target,
                "lastActionSuccess": action_success,
                "removed_from_view": False,
                "target_after": self._summarize_object(target_after),
                "error_message": error_message,
            }

        return {
            "status": "error",
            "result_type": "error_clean_verify_failed",
            "message": "已尝试清扫，但复查后目标仍然可见，清扫未完成。",
            "target": target,
            "lastActionSuccess": action_success,
            "removed_from_view": False,
            "target_after": self._summarize_object(target_after),
            "error_message": error_message,
        }

    def _get_visible_trash_candidates(self) -> List[JsonDict]:
        candidates: List[JsonDict] = []
        for obj in self.last_event.metadata.get("objects", []):
            if obj.get("objectType") not in self.cleanable_object_types:
                continue
            if not obj.get("visible", False):
                continue

            summary = self._summarize_candidate_object(obj)
            candidates.append(summary)

        candidates.sort(key=lambda c: c.get("ground_distance", float("inf")))
        return candidates

    def _summarize_candidate_object(self, obj: JsonDict) -> JsonDict:
        summary = self._summarize_object(obj)
        position_hint = self._relative_position_hint(obj)
        angle_delta = self._relative_angle_delta(obj)
        metadata_distance = float(obj.get("distance", float("inf")))
        ground_distance = self._ground_distance_to_object(obj)
        is_floor_level = self._is_floor_level_object(obj)
        is_near = ground_distance <= float(os.getenv("ROBOT_CLEAN_MAX_DISTANCE", "0.8"))
        is_front_center = position_hint == "front-center"
        clean_rule_passed = is_floor_level and is_near and is_front_center
        summary.update(
            {
                "metadata_distance": round(metadata_distance, 3),
                "distance": round(ground_distance, 3),
                "ground_distance": round(ground_distance, 3),
                "position_hint": position_hint,
                "angle_delta_deg": round(angle_delta, 2),
                "is_floor_level": bool(is_floor_level),
                "is_near": bool(is_near),
                "is_front_center": bool(is_front_center),
                "clean_rule_passed": bool(clean_rule_passed),
                "task_semantic_class": "cleanable_floor_trash" if is_floor_level else "non_floor_object",
                "reject_reasons": self._candidate_reject_reasons(
                    is_floor_level=is_floor_level,
                    is_near=is_near,
                    is_front_center=is_front_center,
                ),
            }
        )
        return summary

    def _candidate_reject_reasons(self, *, is_floor_level: bool, is_near: bool, is_front_center: bool) -> List[str]:
        reasons: List[str] = []
        if not is_front_center:
            reasons.append("target_not_centered")
        if not is_near:
            reasons.append("target_too_far")
        if not is_floor_level:
            reasons.append("target_not_floor_level")
        return reasons

    def _infer_clean_reject_type(self, candidates: List[JsonDict]) -> str:
        if not candidates:
            return "error_no_cleanable_target"
        reject_reasons = candidates[0].get("reject_reasons", [])
        if "target_not_centered" in reject_reasons:
            return "error_target_not_centered"
        if "target_too_far" in reject_reasons:
            return "error_not_reachable"
        if "target_not_floor_level" in reject_reasons:
            return "error_not_floor_level"
        return "error_no_cleanable_target"

    def _clean_reject_message(self, result_type: str) -> str:
        messages = {
            "error_target_not_centered": "发现垃圾代理物，但目标不在前方中间，应先对位。",
            "error_not_reachable": "发现垃圾代理物，但目标距离过远或不可达，应先靠近。",
            "error_not_floor_level": "发现候选物体，但目标不满足地面高度条件，不允许清扫。",
            "error_no_cleanable_target": "发现候选目标，但不满足清扫准入条件。",
        }
        return messages.get(result_type, "不满足清扫准入条件。")

    # ------------------------------------------------------------------
    # Household service actuators: pickup / place / inventory
    # ------------------------------------------------------------------

    def get_inventory_state(self) -> JsonDict:
        inventory = self.last_event.metadata.get("inventoryObjects", []) or []
        inventory = inventory if isinstance(inventory, list) else []
        return {
            "status": "success",
            "schema_version": 2,
            "result_type": "inventory_state",
            "holding_object": bool(inventory),
            "held_object_count": len(inventory),
            "inventory_objects": [self._summarize_object(obj) for obj in inventory if isinstance(obj, dict)],
        }

    def pickup_object_in_front(
        self,
        visual_candidate: Optional[JsonDict] = None,
        strict_visual_grounding: bool = False,
    ) -> JsonDict:
        visual_candidate = self._normalize_visual_candidate(visual_candidate)
        visual_candidate_received = bool(visual_candidate)
        grounding_policy = (
            "metadata_hidden_visual_candidate"
            if visual_candidate_received or strict_visual_grounding
            else "legacy_metadata_front_candidate"
        )
        inventory = self.get_inventory_state()
        if bool(inventory.get("holding_object", False)):
            return {
                "status": "error",
                "schema_version": 2,
                "result_type": "error_already_holding_object",
                "message": "Robot is already holding an object.",
                "holding_object": True,
                "inventory": inventory,
                "grounding_policy": grounding_policy,
                "visual_grounding_required": bool(strict_visual_grounding),
                "visual_candidate_received": visual_candidate_received,
                "visual_candidate_label": self._visual_candidate_label(visual_candidate),
            }

        candidates = self._get_visible_pickup_candidates()
        if not candidates:
            return {
                "status": "error",
                "schema_version": 2,
                "result_type": "error_no_pickup_target_in_front",
                "message": "No visible pickup target is currently eligible.",
                "holding_object": False,
                "candidates": [],
                "grounding_policy": grounding_policy,
                "visual_grounding_required": bool(strict_visual_grounding),
                "visual_candidate_received": visual_candidate_received,
                "visual_candidate_label": self._visual_candidate_label(visual_candidate),
            }

        if visual_candidate_received or strict_visual_grounding:
            grounded = self._ground_visual_service_candidate(
                candidates,
                visual_candidate=visual_candidate,
                task_semantic_class="pickup_target",
            )
            if grounded is None:
                return {
                    "status": "error",
                    "schema_version": 2,
                    "result_type": "error_visual_pickup_target_not_grounded",
                    "message": "The executor could not ground the visual pickup candidate to a visible AI2-THOR object.",
                    "holding_object": False,
                    "visual_candidate": visual_candidate,
                    "candidates": candidates,
                    "grounding_policy": grounding_policy,
                    "visual_grounding_required": bool(strict_visual_grounding),
                    "visual_candidate_received": visual_candidate_received,
                    "visual_candidate_label": self._visual_candidate_label(visual_candidate),
                }
            candidates = [grounded]

        if visual_candidate_received or strict_visual_grounding:
            eligible = [
                c for c in candidates
                if c.get("pickupable") and c.get("is_near_pick")
            ]
        else:
            eligible = [c for c in candidates if c.get("pickup_rule_passed")]
        if not eligible:
            result_type = self._infer_pick_reject_type(candidates)
            return {
                "status": "error",
                "schema_version": 2,
                "result_type": result_type,
                "message": "Visible pickup targets do not satisfy front/near/pickupable constraints.",
                "holding_object": False,
                "candidates": candidates,
                "visual_candidate": visual_candidate,
                "grounding_policy": grounding_policy,
                "visual_grounding_required": bool(strict_visual_grounding),
                "visual_candidate_received": visual_candidate_received,
                "visual_candidate_label": self._visual_candidate_label(visual_candidate),
            }

        eligible.sort(
            key=lambda c: (
                0 if c.get("visual_front_edge_place_ready") else 1,
                0 if c.get("visual_front_center_override") else 1,
                c.get("ground_distance", float("inf")),
            )
        )
        target = eligible[0]
        event = self.controller.step(
            action="PickupObject",
            objectId=target["objectId"],
            forceAction=True,
        )
        action_success = bool(event.metadata.get("lastActionSuccess", False))
        error_message = event.metadata.get("errorMessage", "")
        self.last_event = self.controller.step(action="Pass")
        inventory_after = self.get_inventory_state()
        success = action_success and bool(inventory_after.get("holding_object", False))
        return {
            "status": "success" if success else "error",
            "schema_version": 2,
            "result_type": "pickup_executed" if success else "error_pickup_failed",
            "message": "Pickup executed." if success else "Pickup action failed or inventory did not change.",
            "target": target,
            "lastActionSuccess": action_success,
            "holding_object": bool(inventory_after.get("holding_object", False)),
            "inventory": inventory_after,
            "error_message": error_message,
            "visual_candidate": visual_candidate,
            "grounding_policy": grounding_policy,
            "visual_grounding_required": bool(strict_visual_grounding),
            "visual_candidate_received": visual_candidate_received,
            "visual_candidate_label": self._visual_candidate_label(visual_candidate),
        }

    def place_held_object(
        self,
        visual_candidate: Optional[JsonDict] = None,
        strict_visual_grounding: bool = False,
    ) -> JsonDict:
        visual_candidate = self._normalize_visual_candidate(visual_candidate)
        visual_candidate_received = bool(visual_candidate)
        failed_candidate_id = self._visual_candidate_id(visual_candidate)
        visual_receptacle_grounding: JsonDict = {}
        grounding_policy = (
            "metadata_hidden_visual_candidate"
            if visual_candidate_received or strict_visual_grounding
            else "legacy_metadata_front_candidate"
        )
        inventory = self.get_inventory_state()
        if not bool(inventory.get("holding_object", False)):
            return {
                "status": "error",
                "schema_version": 2,
                "result_type": "error_no_held_object",
                "message": "Robot is not holding an object.",
                "holding_object": False,
                "inventory": inventory,
                "grounding_policy": grounding_policy,
                "visual_grounding_required": bool(strict_visual_grounding),
                "visual_candidate_received": visual_candidate_received,
                "visual_candidate_label": self._visual_candidate_label(visual_candidate),
            }

        candidates = self._get_visible_receptacle_candidates()
        if not candidates:
            return {
                "status": "error",
                "schema_version": 2,
                "result_type": "error_no_receptacle_in_front",
                "message": "No visible place receptacle is currently eligible.",
                "holding_object": True,
                "candidates": [],
                "grounding_policy": grounding_policy,
                "visual_grounding_required": bool(strict_visual_grounding),
                "visual_candidate_received": visual_candidate_received,
                "visual_candidate_label": self._visual_candidate_label(visual_candidate),
            }

        if visual_candidate_received or strict_visual_grounding:
            grounded = self._ground_visual_service_candidate(
                candidates,
                visual_candidate=visual_candidate,
                task_semantic_class="place_receptacle",
            )
            if grounded is None:
                return {
                    "status": "error",
                    "schema_version": 2,
                    "result_type": "error_visual_receptacle_not_grounded",
                    "message": "The executor could not ground the visual receptacle candidate to a visible AI2-THOR object.",
                    "holding_object": True,
                    "visual_candidate": visual_candidate,
                    "inventory": inventory,
                    "candidates": candidates,
                    "grounding_policy": grounding_policy,
                    "visual_grounding_required": bool(strict_visual_grounding),
                    "visual_candidate_received": visual_candidate_received,
                    "visual_candidate_label": self._visual_candidate_label(visual_candidate),
                }
            visual_candidate_stats = self._visual_bbox_stats(visual_candidate)
            grounding_interactable_guidance = self._object_interactable_navigation_guidance(grounded.get("objectId"))
            visual_receptacle_grounding = self._validate_visual_receptacle_grounding(visual_candidate, grounded)
            if not bool(visual_receptacle_grounding.get("passed", False)):
                front_edge_grounding_override = self._allow_front_edge_interactable_grounding_override(
                    visual_candidate=visual_candidate,
                    receptacle=grounded,
                    grounding=visual_receptacle_grounding,
                    interactable_guidance=grounding_interactable_guidance,
                )
                if front_edge_grounding_override.get("needs_approach"):
                    return {
                        "status": "error",
                        "schema_version": 2,
                        "result_type": "error_receptacle_too_far",
                        "message": "The front-edge receptacle is visible, but the current robot pose is not an AI2-THOR interactable pose for it.",
                        "holding_object": True,
                        "visual_candidate": visual_candidate,
                        "inventory": inventory,
                        "candidates": candidates,
                        "visual_receptacle_grounding": visual_receptacle_grounding,
                        "visual_receptacle_grounding_passed": False,
                        "visual_receptacle_grounding_result_type": visual_receptacle_grounding.get("result_type"),
                        "visual_receptacle_target_instance_ratio": visual_receptacle_grounding.get("target_instance_ratio"),
                        "visual_box_ambiguous": visual_receptacle_grounding.get("box_ambiguous"),
                        "interactable_pose_guidance": grounding_interactable_guidance,
                        "executor_action_hint": grounding_interactable_guidance.get("recommended_action"),
                        "interactable_current_pose": bool(grounding_interactable_guidance.get("current_pose_interactable", False)),
                        "interactable_pose_count": grounding_interactable_guidance.get("interactable_pose_count"),
                        "interactable_distance_bucket": grounding_interactable_guidance.get("distance_bucket"),
                        "interactable_angle_bucket": grounding_interactable_guidance.get("angle_bucket"),
                        "grounding_policy": grounding_policy,
                        "visual_grounding_required": bool(strict_visual_grounding),
                        "visual_candidate_received": visual_candidate_received,
                        "visual_candidate_label": self._visual_candidate_label(visual_candidate),
                    }
                if front_edge_grounding_override.get("passed"):
                    visual_receptacle_grounding = {
                        **visual_receptacle_grounding,
                        "passed": True,
                        "result_type": "visual_receptacle_front_edge_interactable_grounded",
                        "message": "The front-edge receptacle box is accepted because the grounded AI2-THOR receptacle is interactable from the current pose.",
                        "evidence_source": "front_edge_interactable_pose_override",
                        "front_edge_interactable_override": True,
                        "interactable_pose_count": grounding_interactable_guidance.get("interactable_pose_count"),
                    }
                else:
                    return {
                        "status": "error",
                        "schema_version": 2,
                        "result_type": str(
                            visual_receptacle_grounding.get("result_type")
                            or "error_visual_receptacle_not_grounded"
                        ),
                        "message": str(
                            visual_receptacle_grounding.get("message")
                            or "The visual receptacle candidate is not grounded on an actionable instance."
                        ),
                        "holding_object": True,
                        "visual_candidate": visual_candidate,
                        "inventory": inventory,
                        "candidates": candidates,
                        "visual_receptacle_grounding": visual_receptacle_grounding,
                        "visual_receptacle_grounding_passed": False,
                        "visual_receptacle_grounding_result_type": visual_receptacle_grounding.get("result_type"),
                        "visual_receptacle_target_instance_ratio": visual_receptacle_grounding.get("target_instance_ratio"),
                        "visual_box_ambiguous": visual_receptacle_grounding.get("box_ambiguous"),
                        "interactable_pose_guidance": grounding_interactable_guidance,
                        "executor_action_hint": grounding_interactable_guidance.get("recommended_action"),
                        "interactable_current_pose": bool(grounding_interactable_guidance.get("current_pose_interactable", False))
                        if grounding_interactable_guidance
                        else None,
                        "interactable_pose_count": grounding_interactable_guidance.get("interactable_pose_count")
                        if grounding_interactable_guidance
                        else None,
                        "interactable_distance_bucket": grounding_interactable_guidance.get("distance_bucket")
                        if grounding_interactable_guidance
                        else None,
                        "interactable_angle_bucket": grounding_interactable_guidance.get("angle_bucket")
                        if grounding_interactable_guidance
                        else None,
                        "grounding_policy": grounding_policy,
                        "visual_grounding_required": bool(strict_visual_grounding),
                        "visual_candidate_received": visual_candidate_received,
                        "visual_candidate_label": self._visual_candidate_label(visual_candidate),
                    }
            if not bool(visual_receptacle_grounding.get("passed", False)):
                return {
                    "status": "error",
                    "schema_version": 2,
                    "result_type": str(
                        visual_receptacle_grounding.get("result_type")
                        or "error_visual_receptacle_not_grounded"
                    ),
                    "message": str(
                        visual_receptacle_grounding.get("message")
                        or "The visual receptacle candidate is not grounded on an actionable instance."
                    ),
                    "holding_object": True,
                    "visual_candidate": visual_candidate,
                    "inventory": inventory,
                    "candidates": candidates,
                    "visual_receptacle_grounding": visual_receptacle_grounding,
                    "visual_receptacle_grounding_passed": False,
                    "visual_receptacle_grounding_result_type": visual_receptacle_grounding.get("result_type"),
                    "visual_receptacle_target_instance_ratio": visual_receptacle_grounding.get("target_instance_ratio"),
                    "visual_box_ambiguous": visual_receptacle_grounding.get("box_ambiguous"),
                    "grounding_policy": grounding_policy,
                    "visual_grounding_required": bool(strict_visual_grounding),
                    "visual_candidate_received": visual_candidate_received,
                    "visual_candidate_label": self._visual_candidate_label(visual_candidate),
                }
            grounded["visual_receptacle_grounding_passed"] = True
            grounded["visual_receptacle_grounding_result_type"] = visual_receptacle_grounding.get("result_type")
            grounded["visual_receptacle_target_instance_ratio"] = visual_receptacle_grounding.get("target_instance_ratio")
            if (
                not self._is_explicit_surface_place_candidate(visual_candidate)
                and (
                    visual_receptacle_grounding.get("broad_front_receptacle")
                    or visual_candidate_stats.get("broad_front_receptacle")
                )
            ):
                interaction_point = visual_receptacle_grounding.get("interaction_point")
                front_edge_visual_ready = bool(
                    visual_candidate_stats.get("front_edge_receptacle")
                    and visual_receptacle_grounding.get("front_edge_receptacle")
                    and isinstance(interaction_point, dict)
                )
                grounded["visual_front_center_override"] = True
                grounded["visual_front_edge_place_ready"] = bool(front_edge_visual_ready)
                grounded["visual_place_rule_source"] = (
                    "front_edge_visual_interaction_point"
                    if front_edge_visual_ready
                    else "broad_front_metadata_near"
                )
                grounded["place_rule_passed"] = bool(
                    grounded.get("allowed_place_receptacle")
                    and grounded.get("receptacle")
                    and (front_edge_visual_ready or grounded.get("is_near_place"))
                )
            self._apply_grounded_surface_place_rule(
                visual_candidate=visual_candidate,
                receptacle=grounded,
                grounding=visual_receptacle_grounding,
            )
            candidates = [grounded]

        eligible = [c for c in candidates if c.get("place_rule_passed")]
        if not eligible:
            result_type = self._infer_place_reject_type(candidates)
            return {
                "status": "error",
                "schema_version": 2,
                "result_type": result_type,
                "message": "Visible receptacles do not satisfy front/near/receptacle constraints.",
                "holding_object": True,
                "candidates": candidates,
                "visual_candidate": visual_candidate,
                "grounding_policy": grounding_policy,
                "visual_grounding_required": bool(strict_visual_grounding),
                "visual_candidate_received": visual_candidate_received,
                "visual_candidate_label": self._visual_candidate_label(visual_candidate),
            }

        raw_inventory = self.last_event.metadata.get("inventoryObjects", []) or []
        raw_inventory = raw_inventory if isinstance(raw_inventory, list) else []
        held_object_id = None
        if raw_inventory and isinstance(raw_inventory[0], dict):
            held_object_id = raw_inventory[0].get("objectId")
        if not held_object_id:
            return {
                "status": "error",
                "schema_version": 2,
                "result_type": "error_no_held_object_id",
                "message": "Robot inventory is non-empty but no held object id is available.",
                "holding_object": True,
                "inventory": inventory,
                "candidates": candidates,
                "visual_candidate": visual_candidate,
                "grounding_policy": grounding_policy,
                "visual_grounding_required": bool(strict_visual_grounding),
                "visual_candidate_received": visual_candidate_received,
                "visual_candidate_label": self._visual_candidate_label(visual_candidate),
            }

        eligible.sort(
            key=lambda c: (
                0 if c.get("visual_front_edge_place_ready") else 1,
                0 if c.get("visual_front_center_override") else 1,
                c.get("ground_distance", float("inf")),
            )
        )
        event = None
        receptacle = eligible[0]
        attempts: List[JsonDict] = []
        interactable_guidance: JsonDict = {}
        place_force_action = _env_bool("ROBOT_PLACE_FORCE_ACTION", False)
        precheck_interactable_pose = _env_bool("ROBOT_PLACE_PRECHECK_INTERACTABLE_POSE", True)
        grid_clearance_contract = self._has_grid_clearance_contract(visual_candidate)
        exact_grid_target_required = self._grid_exact_target_required(visual_candidate)
        executed_exact_target: Optional[JsonDict] = None
        exact_resolution_detail: JsonDict = {}
        placement_execution_mode = (
            "world_point_exact" if exact_grid_target_required else "screen_xy_near_visual_candidate"
        )

        for candidate in eligible:
            receptacle = candidate
            candidate_guidance = self._object_interactable_navigation_guidance(candidate.get("objectId"))
            if candidate_guidance:
                interactable_guidance = candidate_guidance
                attempts.append(
                    {
                        "mode": "receptacle_interactable_pose_precheck",
                        "receptacle": candidate,
                        "available": bool(candidate_guidance.get("available", False)),
                        "current_pose_interactable": bool(candidate_guidance.get("current_pose_interactable", False)),
                        "recommended_action": candidate_guidance.get("recommended_action"),
                        "reason": candidate_guidance.get("reason"),
                    }
                )
                if (
                    precheck_interactable_pose
                    and bool(candidate_guidance.get("available", False))
                    and not bool(candidate_guidance.get("current_pose_interactable", False))
                ):
                    continue
            if exact_grid_target_required:
                exact_targets, exact_resolution_detail = self._resolve_grid_exact_targets(
                    visual_candidate,
                    candidate,
                )
                attempts.append(
                    {
                        "mode": "world_point_exact_target_resolution",
                        "receptacle": candidate,
                        **exact_resolution_detail,
                    }
                )
                for target_index, target in enumerate(exact_targets, start=1):
                    try:
                        event = self.controller.step(
                            action="PlaceObjectAtPoint",
                            objectId=held_object_id,
                            position=target["action_world_target"],
                        )
                        attempts.append(
                            {
                                "mode": "world_point_exact",
                                "receptacle": candidate,
                                "point_index": target_index,
                                "point_count": len(exact_targets),
                                "resolution_error_m": target.get("resolution_error_m"),
                                "lastActionSuccess": bool(event.metadata.get("lastActionSuccess", False)),
                                "error_message": event.metadata.get("errorMessage", ""),
                            }
                        )
                        if bool(event.metadata.get("lastActionSuccess", False)):
                            executed_exact_target = target
                            placement_execution_mode = "world_point_exact"
                            break
                    except Exception as exc:
                        event = None
                        attempts.append(
                            {
                                "mode": "world_point_exact",
                                "receptacle": candidate,
                                "point_index": target_index,
                                "point_count": len(exact_targets),
                                "lastActionSuccess": False,
                                "error_message": str(exc),
                            }
                        )
                if event is not None and bool(event.metadata.get("lastActionSuccess", False)):
                    break
                continue
            place_points = self._visual_place_points(visual_candidate, candidate)
            if place_points:
                for point_index, (x_norm, y_norm) in enumerate(place_points, start=1):
                    try:
                        event = self.controller.step(
                            action="PutObject",
                            x=x_norm,
                            y=y_norm,
                            forceAction=place_force_action,
                            placeStationary=True,
                            putNearXY=True,
                        )
                        attempts.append(
                            {
                                "mode": "screen_xy_near_visual_candidate",
                                "receptacle": candidate,
                                "point_index": point_index,
                                "point_count": len(place_points),
                                "x": round(x_norm, 3),
                                "y": round(y_norm, 3),
                                "forceAction": bool(place_force_action),
                                "lastActionSuccess": bool(event.metadata.get("lastActionSuccess", False)),
                                "error_message": event.metadata.get("errorMessage", ""),
                            }
                        )
                        if bool(event.metadata.get("lastActionSuccess", False)):
                            break
                    except Exception as exc:
                        event = None
                        attempts.append(
                            {
                                "mode": "screen_xy_near_visual_candidate",
                                "receptacle": candidate,
                                "point_index": point_index,
                                "point_count": len(place_points),
                                "lastActionSuccess": False,
                                "error_message": str(exc),
                            }
                        )
                if event is not None and bool(event.metadata.get("lastActionSuccess", False)):
                    break
            else:
                attempts.append(
                    {
                        "mode": "screen_xy_near_visual_candidate",
                        "receptacle": candidate,
                        "lastActionSuccess": False,
                        "error_message": "no visual placement point available",
                    }
                )

        if event is None:
            if (
                interactable_guidance
                and bool(interactable_guidance.get("available", False))
                and not bool(interactable_guidance.get("current_pose_interactable", False))
            ):
                return {
                    "status": "error",
                    "schema_version": 2,
                    "result_type": "error_place_pose_not_interactable",
                    "message": "The grounded receptacle is visible, but the current robot pose cannot execute placement on it.",
                    "holding_object": True,
                    "inventory": inventory,
                    "candidates": candidates,
                    "attempts": attempts,
                    "interactable_pose_guidance": interactable_guidance,
                    "executor_action_hint": interactable_guidance.get("recommended_action"),
                    "interactable_current_pose": bool(interactable_guidance.get("current_pose_interactable", False)),
                    "interactable_pose_count": interactable_guidance.get("interactable_pose_count"),
                    "interactable_distance_bucket": interactable_guidance.get("distance_bucket"),
                    "interactable_angle_bucket": interactable_guidance.get("angle_bucket"),
                    "visual_candidate": visual_candidate,
                    "grounding_policy": grounding_policy,
                    "visual_grounding_required": bool(strict_visual_grounding),
                    "visual_candidate_received": visual_candidate_received,
                    "visual_candidate_label": self._visual_candidate_label(visual_candidate),
                    "candidate_id": failed_candidate_id,
                    "failed_candidate_id": failed_candidate_id,
                    "placement_point_source": (
                        "pointcloud_plane_local_grid" if grid_clearance_contract else "executor_visual_clearance"
                    ),
                    "placement_clearance_contract_applied": bool(grid_clearance_contract),
                    "placement_failure_stage": "actuator_pose",
                    "placement_execution_mode": placement_execution_mode,
                    "placement_target_required": bool(exact_grid_target_required),
                }
            if exact_grid_target_required:
                return {
                    "status": "error",
                    "schema_version": 2,
                    "result_type": "error_place_exact_target_unavailable",
                    "message": "No simulator-legal exact placement point remained near the RGB-D selected target.",
                    "holding_object": True,
                    "inventory": inventory,
                    "candidates": candidates,
                    "attempts": attempts,
                    "visual_candidate": visual_candidate,
                    "grounding_policy": grounding_policy,
                    "visual_grounding_required": bool(strict_visual_grounding),
                    "visual_candidate_received": visual_candidate_received,
                    "visual_candidate_label": self._visual_candidate_label(visual_candidate),
                    "candidate_id": failed_candidate_id,
                    "failed_candidate_id": failed_candidate_id,
                    "placement_point_source": "pointcloud_plane_local_grid",
                    "placement_clearance_contract_applied": True,
                    "placement_failure_stage": "exact_target_resolution",
                    "placement_execution_mode": "world_point_exact",
                    "placement_target_required": True,
                    "placement_target_resolution_error_m": exact_resolution_detail.get("best_resolution_error_m"),
                    "placement_target_resolution_tolerance_m": exact_resolution_detail.get("max_resolution_error_m"),
                }
            result_type = "error_place_point_clearance" if grid_clearance_contract else "error_place_no_reachable_point"
            return {
                "status": "error",
                "schema_version": 2,
                "result_type": result_type,
                "message": (
                    "No point from the pointcloud placement safety contract remained executable."
                    if grid_clearance_contract
                    else "No controlled reachable placement point was available."
                ),
                "holding_object": True,
                "inventory": inventory,
                "candidates": candidates,
                "attempts": attempts,
                "visual_candidate": visual_candidate,
                "grounding_policy": grounding_policy,
                "visual_grounding_required": bool(strict_visual_grounding),
                "visual_candidate_received": visual_candidate_received,
                "visual_candidate_label": self._visual_candidate_label(visual_candidate),
                "candidate_id": failed_candidate_id,
                "failed_candidate_id": failed_candidate_id,
                "placement_point_source": (
                    "pointcloud_plane_local_grid" if grid_clearance_contract else "executor_visual_clearance"
                ),
                "placement_clearance_contract_applied": bool(grid_clearance_contract),
                "placement_failure_stage": (
                    "pointcloud_safe_points_exhausted"
                    if grid_clearance_contract
                    else "executor_visual_point_search"
                ),
                "placement_execution_mode": placement_execution_mode,
                "placement_target_required": bool(exact_grid_target_required),
            }
        action_success = bool(event.metadata.get("lastActionSuccess", False))
        error_message = event.metadata.get("errorMessage", "")
        self.last_event = self.controller.step(action="Pass")
        inventory_after = self.get_inventory_state()
        inventory_empty = not bool(inventory_after.get("holding_object", False))
        placement_validation = self._validate_place_result(
            held_object_id=held_object_id,
            receptacle=receptacle,
            action_success=action_success,
            inventory_empty=inventory_empty,
            expected_surface_world_target=(
                executed_exact_target.get("surface_world_target")
                if isinstance(executed_exact_target, dict)
                else None
            ),
            exact_target_required=bool(executed_exact_target),
        )
        success = action_success and inventory_empty and bool(placement_validation.get("placement_verified", False))
        result_type = "place_executed" if success else str(placement_validation.get("result_type") or "error_place_failed")
        return {
            "status": "success" if success else "error",
            "schema_version": 2,
            "result_type": result_type,
            "message": (
                (
                    "Place executed and verified at the selected RGB-D target."
                    if executed_exact_target
                    else "Place executed and verified reachable."
                )
                if success
                else str(placement_validation.get("message") or "Place action failed or placement could not be verified.")
            ),
            "receptacle": receptacle,
            "lastActionSuccess": action_success,
            "holding_object": bool(inventory_after.get("holding_object", False)),
            "inventory": inventory_after,
            "attempts": attempts,
            "placement_validation": placement_validation,
            "placement_verified": bool(placement_validation.get("placement_verified", False)),
            "object_on_target_receptacle": bool(placement_validation.get("object_on_target_receptacle", False)),
            "object_reachable_from_agent": bool(placement_validation.get("object_reachable_from_agent", False)),
            "placement_distance_bucket": placement_validation.get("distance_bucket"),
            "placement_angle_bucket": placement_validation.get("angle_bucket"),
            "interactable_pose_guidance": interactable_guidance,
            "executor_action_hint": interactable_guidance.get("recommended_action") if interactable_guidance else None,
            "interactable_current_pose": bool(interactable_guidance.get("current_pose_interactable", False)) if interactable_guidance else None,
            "interactable_pose_count": interactable_guidance.get("interactable_pose_count") if interactable_guidance else None,
            "interactable_distance_bucket": interactable_guidance.get("distance_bucket") if interactable_guidance else None,
            "interactable_angle_bucket": interactable_guidance.get("angle_bucket") if interactable_guidance else None,
            "visual_receptacle_grounding": visual_receptacle_grounding,
            "visual_receptacle_grounding_passed": bool(visual_receptacle_grounding.get("passed", False)),
            "visual_receptacle_grounding_result_type": visual_receptacle_grounding.get("result_type"),
            "visual_receptacle_target_instance_ratio": visual_receptacle_grounding.get("target_instance_ratio"),
            "visual_box_ambiguous": visual_receptacle_grounding.get("box_ambiguous"),
            "error_message": error_message,
            "visual_candidate": visual_candidate,
            "grounding_policy": grounding_policy,
            "visual_grounding_required": bool(strict_visual_grounding),
            "visual_candidate_received": visual_candidate_received,
            "visual_candidate_label": self._visual_candidate_label(visual_candidate),
            "placement_point_source": (
                "pointcloud_plane_local_grid" if grid_clearance_contract else "executor_visual_clearance"
            ),
            "placement_clearance_contract_applied": bool(grid_clearance_contract),
            "placement_execution_mode": placement_execution_mode,
            "placement_target_required": bool(exact_grid_target_required),
            "placement_target_verified": bool(placement_validation.get("placement_target_verified", False)),
            "placement_target_error_m": placement_validation.get("placement_target_error_m"),
            "placement_target_tolerance_m": placement_validation.get("placement_target_tolerance_m"),
            "placement_target_resolution_error_m": (
                executed_exact_target.get("resolution_error_m")
                if isinstance(executed_exact_target, dict)
                else exact_resolution_detail.get("best_resolution_error_m")
            ),
            "placement_target_resolution_tolerance_m": exact_resolution_detail.get("max_resolution_error_m"),
            "placement_failure_stage": (
                None
                if success
                else "exact_target_validation"
                if executed_exact_target
                else "exact_world_point_action"
                if exact_grid_target_required
                else None
            ),
        }

    def precheck_place_candidate(
        self,
        visual_candidate: Optional[JsonDict] = None,
        strict_visual_grounding: bool = False,
    ) -> JsonDict:
        """Online-safe dry check for whether the current pose can execute placement."""
        visual_candidate = self._normalize_visual_candidate(visual_candidate)
        visual_candidate_received = bool(visual_candidate)
        failed_candidate_id = self._visual_candidate_id(visual_candidate)
        grounding_policy = (
            "metadata_hidden_visual_candidate"
            if visual_candidate_received or strict_visual_grounding
            else "legacy_metadata_front_candidate"
        )

        def result(
            *,
            status: str,
            result_type: str,
            message: str,
            precheck_ok: bool,
            suggested_recovery: Optional[str] = None,
            **extra: Any,
        ) -> JsonDict:
            payload: JsonDict = {
                "status": status,
                "schema_version": 2,
                "result_type": result_type,
                "message": message,
                "precheck_supported": True,
                "precheck_ok": bool(precheck_ok),
                "precheck_reason": result_type,
                "suggested_recovery": suggested_recovery,
                "candidate_id": failed_candidate_id,
                "failed_candidate_id": failed_candidate_id,
                "grounding_policy": grounding_policy,
                "visual_grounding_required": bool(strict_visual_grounding),
                "visual_candidate_received": visual_candidate_received,
                "visual_candidate_label": self._visual_candidate_label(visual_candidate),
                "failed_candidate_id": failed_candidate_id,
            }
            payload.update(extra)
            return payload

        inventory = self.get_inventory_state()
        if not bool(inventory.get("holding_object", False)):
            return result(
                status="error",
                result_type="error_no_held_object",
                message="Robot is not holding an object.",
                precheck_ok=False,
                suggested_recovery="search_pickup_target",
                holding_object=False,
                inventory=inventory,
            )

        candidates = self._get_visible_receptacle_candidates()
        if not candidates:
            return result(
                status="error",
                result_type="error_no_receptacle_in_front",
                message="No visible place receptacle is currently eligible.",
                precheck_ok=False,
                suggested_recovery="search_receptacle",
                holding_object=True,
                inventory=inventory,
                candidates=[],
            )

        visual_receptacle_grounding: JsonDict = {}
        if visual_candidate_received or strict_visual_grounding:
            grounded = self._ground_visual_service_candidate(
                candidates,
                visual_candidate=visual_candidate,
                task_semantic_class="place_receptacle",
            )
            if grounded is None:
                return result(
                    status="error",
                    result_type="error_visual_receptacle_not_grounded",
                    message="The executor could not ground the visual receptacle candidate to a visible AI2-THOR object.",
                    precheck_ok=False,
                    suggested_recovery="search_receptacle",
                    holding_object=True,
                    inventory=inventory,
                    candidates=candidates,
                )
            guidance = self._object_interactable_navigation_guidance(grounded.get("objectId"))
            visual_receptacle_grounding = self._validate_visual_receptacle_grounding(visual_candidate, grounded)
            if not bool(visual_receptacle_grounding.get("passed", False)):
                override = self._allow_front_edge_interactable_grounding_override(
                    visual_candidate=visual_candidate,
                    receptacle=grounded,
                    grounding=visual_receptacle_grounding,
                    interactable_guidance=guidance,
                )
                if override.get("passed"):
                    visual_receptacle_grounding = {
                        **visual_receptacle_grounding,
                        "passed": True,
                        "result_type": "visual_receptacle_front_edge_interactable_grounded",
                    }
                else:
                    suggested = guidance.get("recommended_action") if isinstance(guidance, dict) else None
                    return result(
                        status="error",
                        result_type=str(
                            visual_receptacle_grounding.get("result_type")
                            or "error_visual_receptacle_not_grounded"
                        ),
                        message=str(
                            visual_receptacle_grounding.get("message")
                            or "The visual receptacle candidate is not grounded on an actionable instance."
                        ),
                        precheck_ok=False,
                        suggested_recovery=str(suggested or "search_receptacle"),
                        holding_object=True,
                        inventory=inventory,
                        visual_receptacle_grounding=visual_receptacle_grounding,
                        visual_receptacle_grounding_passed=False,
                        visual_receptacle_grounding_result_type=visual_receptacle_grounding.get("result_type"),
                        visual_box_ambiguous=visual_receptacle_grounding.get("box_ambiguous"),
                        executor_action_hint=guidance.get("recommended_action") if isinstance(guidance, dict) else None,
                        interactable_current_pose=bool(guidance.get("current_pose_interactable", False))
                        if isinstance(guidance, dict)
                        else None,
                        interactable_pose_count=guidance.get("interactable_pose_count") if isinstance(guidance, dict) else None,
                        interactable_distance_bucket=guidance.get("distance_bucket") if isinstance(guidance, dict) else None,
                        interactable_angle_bucket=guidance.get("angle_bucket") if isinstance(guidance, dict) else None,
                    )
            grounded["visual_receptacle_grounding_passed"] = True
            grounded["visual_receptacle_grounding_result_type"] = visual_receptacle_grounding.get("result_type")
            self._apply_grounded_surface_place_rule(
                visual_candidate=visual_candidate,
                receptacle=grounded,
                grounding=visual_receptacle_grounding,
            )
            candidates = [grounded]

        eligible = [c for c in candidates if c.get("place_rule_passed")]
        if not eligible:
            result_type = self._infer_place_reject_type(candidates)
            return result(
                status="error",
                result_type=result_type,
                message="Visible receptacles do not satisfy front/near/receptacle constraints.",
                precheck_ok=False,
                suggested_recovery="align" if result_type == "error_receptacle_not_centered" else "approach",
                holding_object=True,
                inventory=inventory,
                candidates=candidates,
            )

        eligible.sort(
            key=lambda c: (
                0 if c.get("visual_front_edge_place_ready") else 1,
                0 if c.get("visual_front_center_override") else 1,
                c.get("ground_distance", float("inf")),
            )
        )
        attempts: List[JsonDict] = []
        grid_clearance_contract = self._has_grid_clearance_contract(visual_candidate)
        exact_grid_target_required = self._grid_exact_target_required(visual_candidate)
        exact_resolution_detail: JsonDict = {}
        for candidate in eligible:
            guidance = self._object_interactable_navigation_guidance(candidate.get("objectId"))
            if (
                _env_bool("ROBOT_PLACE_PRECHECK_INTERACTABLE_POSE", True)
                and bool(guidance.get("available", False))
                and not bool(guidance.get("current_pose_interactable", False))
            ):
                attempts.append(
                    {
                        "mode": "receptacle_interactable_pose_precheck",
                        "available": bool(guidance.get("available", False)),
                        "current_pose_interactable": False,
                        "recommended_action": guidance.get("recommended_action"),
                        "reason": guidance.get("reason"),
                    }
                )
                continue
            if exact_grid_target_required:
                exact_targets, exact_resolution_detail = self._resolve_grid_exact_targets(
                    visual_candidate,
                    candidate,
                )
                attempts.append(
                    {
                        "mode": "world_point_exact_target_resolution",
                        **exact_resolution_detail,
                    }
                )
                if exact_targets:
                    return result(
                        status="success",
                        result_type="place_precheck_ok",
                        message="Visual surface candidate has a grounded receptacle and an exact world placement target.",
                        precheck_ok=True,
                        suggested_recovery=None,
                        holding_object=True,
                        inventory=inventory,
                        visual_receptacle_grounding=visual_receptacle_grounding,
                        visual_receptacle_grounding_passed=bool(visual_receptacle_grounding.get("passed", False)),
                        visual_receptacle_grounding_result_type=visual_receptacle_grounding.get("result_type"),
                        placement_attempt_count=len(exact_targets),
                        placement_attempt_modes=["world_point_exact"],
                        placement_point_source="pointcloud_plane_local_grid",
                        placement_clearance_contract_applied=True,
                        placement_execution_mode="world_point_exact",
                        placement_target_required=True,
                        placement_target_resolution_error_m=exact_targets[0].get("resolution_error_m"),
                        placement_target_resolution_tolerance_m=exact_resolution_detail.get("max_resolution_error_m"),
                        interactable_current_pose=bool(guidance.get("current_pose_interactable", False)) if guidance else None,
                        interactable_pose_count=guidance.get("interactable_pose_count") if guidance else None,
                        interactable_distance_bucket=guidance.get("distance_bucket") if guidance else None,
                        interactable_angle_bucket=guidance.get("angle_bucket") if guidance else None,
                    )
                continue
            place_points = self._visual_place_points(visual_candidate, candidate)
            attempts.append(
                {
                    "mode": "screen_xy_near_visual_candidate",
                    "point_count": len(place_points),
                    "interactable_current_pose": bool(guidance.get("current_pose_interactable", False))
                    if guidance
                    else None,
                }
            )
            if place_points:
                return result(
                    status="success",
                    result_type="place_precheck_ok",
                    message="Visual surface candidate has a grounded receptacle and controlled screen placement point.",
                    precheck_ok=True,
                    suggested_recovery=None,
                    holding_object=True,
                    inventory=inventory,
                    visual_receptacle_grounding=visual_receptacle_grounding,
                    visual_receptacle_grounding_passed=bool(visual_receptacle_grounding.get("passed", False)),
                    visual_receptacle_grounding_result_type=visual_receptacle_grounding.get("result_type"),
                    placement_attempt_count=len(place_points),
                    placement_attempt_modes=["screen_xy_near_visual_candidate"],
                    placement_point_source=(
                        "pointcloud_plane_local_grid" if grid_clearance_contract else "executor_visual_clearance"
                    ),
                    placement_clearance_contract_applied=bool(grid_clearance_contract),
                    interactable_current_pose=bool(guidance.get("current_pose_interactable", False)) if guidance else None,
                    interactable_pose_count=guidance.get("interactable_pose_count") if guidance else None,
                    interactable_distance_bucket=guidance.get("distance_bucket") if guidance else None,
                    interactable_angle_bucket=guidance.get("angle_bucket") if guidance else None,
                )

        suggested = None
        for attempt in attempts:
            if attempt.get("recommended_action"):
                suggested = str(attempt.get("recommended_action"))
                break
        pose_rejected = any(
            attempt.get("mode") == "receptacle_interactable_pose_precheck"
            and attempt.get("available")
            and not attempt.get("current_pose_interactable")
            for attempt in attempts
        )
        point_evaluated = any(attempt.get("mode") == "screen_xy_near_visual_candidate" for attempt in attempts)
        if pose_rejected and not point_evaluated:
            result_type = "error_place_pose_not_interactable"
            failure_stage = "actuator_pose"
            message = "The grounded receptacle is not interactable from the current robot pose."
        elif exact_grid_target_required:
            result_type = "error_place_exact_target_unavailable"
            failure_stage = "exact_target_resolution"
            message = "No simulator-legal exact placement point is close enough to the RGB-D selected target."
        elif grid_clearance_contract:
            result_type = "error_place_point_clearance"
            failure_stage = "pointcloud_safe_points_exhausted"
            message = "No point from the pointcloud placement safety contract is available for execution."
        else:
            result_type = "error_place_no_reachable_point"
            failure_stage = "executor_visual_point_search"
            message = "No controlled reachable placement point is available for this surface candidate."
        return result(
            status="error",
            result_type=result_type,
            message=message,
            precheck_ok=False,
            suggested_recovery=suggested or "search_receptacle",
            holding_object=True,
            inventory=inventory,
            attempts=attempts,
            placement_attempt_count=0,
            placement_attempt_modes=["screen_xy_near_visual_candidate"],
            placement_point_source=(
                "pointcloud_plane_local_grid" if grid_clearance_contract else "executor_visual_clearance"
            ),
            placement_clearance_contract_applied=bool(grid_clearance_contract),
            precheck_failure_stage=failure_stage,
            executor_action_hint=suggested,
            placement_execution_mode="world_point_exact" if exact_grid_target_required else "screen_xy_near_visual_candidate",
            placement_target_required=bool(exact_grid_target_required),
            placement_target_resolution_error_m=exact_resolution_detail.get("best_resolution_error_m"),
            placement_target_resolution_tolerance_m=exact_resolution_detail.get("max_resolution_error_m"),
        )

    def _normalize_visual_candidate(self, candidate: Optional[JsonDict]) -> JsonDict:
        if not isinstance(candidate, dict):
            return {}
        safe: JsonDict = {}
        for key in (
            "schema_version",
            "role",
            "id",
            "surface_candidate_id",
            "label",
            "raw_label",
            "task_semantic_class",
            "region_type",
            "region_bbox",
            "region_area_px",
            "region_area_ratio",
            "parent_object",
            "parent_label",
            "source",
            "surface_candidate_source",
            "confidence",
            "position_hint",
            "surface_hint",
            "floor_level_source",
            "projected_height_warning",
            "context_only",
            "context_reason",
            "blocked",
            "score",
            "height",
            "distance",
            "ground_distance",
            "height_m",
            "distance_m",
            "bearing_deg",
            "reachable",
            "pickup_now",
            "place_now",
            "visual_place_ready",
            "affordance_ready",
            "final_place_ready",
            "failed_recently",
            "cooldown_remaining",
            "needs_alignment",
            "needs_approach",
            "is_floor_level",
            "is_support_surface",
            "visual_box_ambiguous",
            "broad_front_receptacle",
            "front_edge_receptacle",
            "area_ratio",
            "center_y_ratio",
            "bottom_y_ratio",
        ):
            if key in candidate:
                safe[key] = candidate.get(key)
        for key in (
            "bbox",
            "center",
            "interaction_point",
            "geometry",
            "depth",
            "center_3d",
            "parent_bbox",
            "region_bbox",
            "geometry_checks",
            "occupancy_checks",
            "memory_checks",
            "executor_checks",
            "free_space_completion",
            "placement_safety_contract",
        ):
            value = candidate.get(key)
            if isinstance(value, dict):
                safe[key] = dict(value)
        for key in ("visible_occupants", "placement_avoidance_candidates", "placement_points", "blocked_by", "affordance", "rejection_reasons"):
            value = candidate.get(key)
            if isinstance(value, list):
                if key in {"blocked_by", "affordance", "rejection_reasons"}:
                    safe[key] = [str(item) for item in value]
                else:
                    safe[key] = [dict(item) for item in value if isinstance(item, dict)]
        return safe

    def _visual_candidate_label(self, candidate: JsonDict) -> Optional[str]:
        if not isinstance(candidate, dict):
            return None
        label = candidate.get("raw_label") or candidate.get("label")
        return str(label) if label else None

    def _visual_candidate_id(self, candidate: JsonDict) -> Optional[str]:
        if not isinstance(candidate, dict):
            return None
        for key in ("failed_candidate_id", "surface_candidate_id", "id"):
            value = candidate.get(key)
            if value:
                return str(value)
        return None

    def _label_token(self, value: Any) -> str:
        text = str(value or "").lower()
        return "".join(ch for ch in text if ch.isalnum())

    def _visual_label_tokens(self, candidate: JsonDict) -> Set[str]:
        labels = {
            self._label_token(candidate.get("raw_label")),
            self._label_token(candidate.get("label")),
        }
        return {label for label in labels if label}

    def _visual_bbox(self, candidate: JsonDict) -> Optional[Tuple[float, float, float, float]]:
        bbox = candidate.get("bbox") if isinstance(candidate.get("bbox"), dict) else {}
        try:
            x = float(bbox.get("x"))
            y = float(bbox.get("y"))
            w = float(bbox.get("w"))
            h = float(bbox.get("h"))
        except (TypeError, ValueError):
            return None
        if w <= 0 or h <= 0:
            return None
        return (x, y, x + w, y + h)

    def _visual_point(self, candidate: JsonDict) -> Optional[Tuple[float, float]]:
        for key in ("interaction_point", "center"):
            point = candidate.get(key) if isinstance(candidate.get(key), dict) else {}
            try:
                x = float(point.get("x"))
                y = float(point.get("y"))
            except (TypeError, ValueError):
                continue
            return (x, y)
        bbox = self._visual_bbox(candidate)
        if bbox is None:
            return None
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _detection_bbox_for_object_id(self, object_id: Any) -> Optional[Tuple[float, float, float, float]]:
        detections = self._instance_detections_2d()
        raw = detections.get(str(object_id)) if isinstance(detections, dict) else None
        if raw is None:
            return None
        try:
            values = raw.tolist() if hasattr(raw, "tolist") else list(raw)
            if len(values) < 4:
                return None
            x1, y1, x2, y2 = [float(v) for v in values[:4]]
        except (TypeError, ValueError):
            return None
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return (x1, y1, x2, y2)

    def _bbox_iou(self, a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter
        return inter / denom if denom > 0 else 0.0

    def _point_in_bbox(self, point: Tuple[float, float], bbox: Tuple[float, float, float, float]) -> bool:
        x, y = point
        x1, y1, x2, y2 = bbox
        return x1 <= x <= x2 and y1 <= y <= y2

    def _visual_bbox_stats(self, candidate: JsonDict) -> JsonDict:
        bbox = self._visual_bbox(candidate)
        if bbox is None:
            return {
                "has_bbox": False,
                "area_ratio": 0.0,
                "width_ratio": 0.0,
                "height_ratio": 0.0,
                "box_ambiguous": False,
            }
        x1, y1, x2, y2 = bbox
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        area_ratio = (width * height) / max(1.0, float(self.width * self.height))
        width_ratio = width / max(1.0, float(self.width))
        height_ratio = height / max(1.0, float(self.height))
        max_area = _env_float("ROBOT_PLACE_MAX_AREA", 0.45)
        max_width = _env_float("ROBOT_PLACE_MAX_WIDTH", 0.995)
        geometry = candidate.get("geometry") if isinstance(candidate.get("geometry"), dict) else {}
        try:
            cx_ratio = float(geometry.get("cx_ratio", 0.5) or 0.5)
        except (TypeError, ValueError):
            cx_ratio = 0.5
        try:
            bottom_y_ratio = float(geometry.get("bottom_y_ratio", candidate.get("bottom_y_ratio", 0.0)) or 0.0)
        except (TypeError, ValueError):
            bottom_y_ratio = 0.0
        try:
            center_y_ratio = float(geometry.get("cy_ratio", candidate.get("center_y_ratio", 0.0)) or 0.0)
        except (TypeError, ValueError):
            center_y_ratio = 0.0
        candidate_broad_front = candidate.get("broad_front_receptacle")
        if isinstance(candidate_broad_front, str):
            candidate_broad_front = candidate_broad_front.strip().lower() in {"1", "true", "yes", "y"}
        candidate_front_edge = candidate.get("front_edge_receptacle")
        if isinstance(candidate_front_edge, str):
            candidate_front_edge = candidate_front_edge.strip().lower() in {"1", "true", "yes", "y"}
        front_edge_receptacle = bool(
            candidate_front_edge
            or (
                str(candidate.get("task_semantic_class") or "") == "place_receptacle"
                and str(candidate.get("position_hint") or "") == "front-center"
                and abs(cx_ratio - 0.5) <= 0.10
                and bottom_y_ratio >= _env_float("ROBOT_PLACE_FRONT_EDGE_MIN_BOTTOM_RATIO", 0.88)
                and center_y_ratio >= _env_float("ROBOT_PLACE_FRONT_EDGE_MIN_CENTER_Y_RATIO", 0.58)
                and area_ratio <= _env_float("ROBOT_PLACE_FRONT_EDGE_MAX_AREA_RATIO", 0.72)
                and width_ratio <= _env_float("ROBOT_PLACE_FRONT_EDGE_MAX_WIDTH_RATIO", 1.01)
            )
        )
        broad_front_receptacle = bool(
            candidate_broad_front
            or front_edge_receptacle
            or (
                str(candidate.get("task_semantic_class") or "") == "place_receptacle"
                and str(candidate.get("position_hint") or "") == "front-center"
                and abs(cx_ratio - 0.5) <= 0.10
                and bottom_y_ratio >= 0.66
                and area_ratio <= max_area
            )
        )
        return {
            "has_bbox": True,
            "area_ratio": round(area_ratio, 6),
            "width_ratio": round(width_ratio, 6),
            "height_ratio": round(height_ratio, 6),
            "box_ambiguous": bool(
                (area_ratio > max_area or width_ratio > max_width)
                and not broad_front_receptacle
            ),
            "broad_front_receptacle": bool(broad_front_receptacle),
            "front_edge_receptacle": bool(front_edge_receptacle),
            "max_area_ratio": round(max_area, 6),
            "max_width_ratio": round(max_width, 6),
        }

    def _held_object_ids(self) -> Set[str]:
        inventory = self.last_event.metadata.get("inventoryObjects", []) or []
        if not isinstance(inventory, list):
            return set()
        ids: Set[str] = set()
        for obj in inventory:
            if isinstance(obj, dict) and obj.get("objectId"):
                ids.add(str(obj.get("objectId")))
        return ids

    def _frame_rgb_at(self, frame: Any, x: int, y: int) -> Optional[Tuple[int, int, int]]:
        try:
            pixel = frame[y, x]
        except Exception:
            try:
                pixel = frame[y][x]
            except Exception:
                return None
        return self._normalize_rgb_color(pixel)

    def _instance_patch_evidence(
        self,
        *,
        target_object_id: Any,
        point_norm: Tuple[float, float],
    ) -> JsonDict:
        frame = getattr(self.last_event, "instance_segmentation_frame", None)
        color_map = self._instance_color_to_object_id()
        if frame is None or not color_map:
            return {"segmentation_available": False, "target_instance_ratio": 0.0, "target_instance_pixels": 0}

        try:
            shape = getattr(frame, "shape", None)
            height = int(shape[0]) if shape is not None and len(shape) >= 2 else int(self.height)
            width = int(shape[1]) if shape is not None and len(shape) >= 2 else int(self.width)
        except Exception:
            height = int(self.height)
            width = int(self.width)

        x_norm, y_norm = point_norm
        cx = int(round(min(0.99, max(0.0, x_norm)) * max(1, width - 1)))
        cy = int(round(min(0.99, max(0.0, y_norm)) * max(1, height - 1)))
        radius = max(4, int(_env_float("ROBOT_PLACE_VISUAL_PATCH_RADIUS", 24.0)))
        step = max(1, int(_env_float("ROBOT_PLACE_VISUAL_PATCH_STEP", 4.0)))
        held_ids = self._held_object_ids()
        target_id = str(target_object_id or "")

        counts: Dict[str, int] = {}
        sampled_pixels = 0
        mapped_pixels = 0
        held_pixels = 0
        for y in range(max(0, cy - radius), min(height, cy + radius + 1), step):
            for x in range(max(0, cx - radius), min(width, cx + radius + 1), step):
                sampled_pixels += 1
                rgb = self._frame_rgb_at(frame, x, y)
                if rgb is None:
                    continue
                object_id = color_map.get(rgb)
                if not object_id:
                    continue
                mapped_pixels += 1
                object_id = str(object_id)
                if object_id in held_ids:
                    held_pixels += 1
                    continue
                counts[object_id] = counts.get(object_id, 0) + 1

        counted_pixels = sum(counts.values())
        target_pixels = int(counts.get(target_id, 0))
        target_ratio = target_pixels / counted_pixels if counted_pixels > 0 else 0.0
        dominant_id = None
        dominant_pixels = 0
        for object_id, count in counts.items():
            if count > dominant_pixels:
                dominant_id = object_id
                dominant_pixels = count

        dominant_type = None
        if dominant_id:
            obj = self._find_object_by_id(dominant_id)
            dominant_type = obj.get("objectType") if isinstance(obj, dict) else None

        return {
            "segmentation_available": True,
            "sampled_pixels": sampled_pixels,
            "mapped_pixels": mapped_pixels,
            "counted_pixels": counted_pixels,
            "held_object_pixels": held_pixels,
            "target_instance_pixels": target_pixels,
            "target_instance_ratio": round(target_ratio, 4),
            "dominant_instance_pixels": dominant_pixels,
            "dominant_instance_type": dominant_type,
            "target_is_dominant": bool(dominant_id == target_id and target_pixels > 0),
        }

    def _instance_bbox_evidence(
        self,
        *,
        target_object_id: Any,
        visual_candidate: JsonDict,
    ) -> JsonDict:
        frame = getattr(self.last_event, "instance_segmentation_frame", None)
        color_map = self._instance_color_to_object_id()
        bbox = self._visual_bbox(visual_candidate)
        if frame is None or not color_map or bbox is None:
            return {
                "segmentation_available": False,
                "target_instance_ratio": 0.0,
                "target_instance_pixels": 0,
                "selected_point_norm": None,
            }

        try:
            shape = getattr(frame, "shape", None)
            height = int(shape[0]) if shape is not None and len(shape) >= 2 else int(self.height)
            width = int(shape[1]) if shape is not None and len(shape) >= 2 else int(self.width)
        except Exception:
            height = int(self.height)
            width = int(self.width)

        x1, y1, x2, y2 = bbox
        min_x = max(0, min(width - 1, int(math.floor(x1))))
        max_x = max(0, min(width - 1, int(math.ceil(x2))))
        min_y = max(0, min(height - 1, int(math.floor(y1))))
        max_y = max(0, min(height - 1, int(math.ceil(y2))))
        if max_x <= min_x or max_y <= min_y:
            return {
                "segmentation_available": True,
                "target_instance_ratio": 0.0,
                "target_instance_pixels": 0,
                "selected_point_norm": None,
            }

        stats = self._visual_bbox_stats(visual_candidate)
        y_ratio = _env_float(
            "ROBOT_PLACE_BROAD_POINT_BBOX_Y_RATIO" if stats.get("broad_front_receptacle") else "ROBOT_PLACE_POINT_BBOX_Y_RATIO",
            0.62 if stats.get("broad_front_receptacle") else 0.88,
        )
        y_ratio = min(0.92, max(0.35, y_ratio))
        preferred_x = (x1 + x2) / 2.0
        preferred_y = y1 + (y2 - y1) * y_ratio

        step = max(1, int(_env_float("ROBOT_PLACE_VISUAL_BBOX_STEP", 3.0)))
        held_ids = self._held_object_ids()
        target_id = str(target_object_id or "")
        counts: Dict[str, int] = {}
        sampled_pixels = 0
        mapped_pixels = 0
        held_pixels = 0
        best_point: Optional[Tuple[int, int]] = None
        best_distance = float("inf")

        for y in range(min_y, max_y + 1, step):
            for x in range(min_x, max_x + 1, step):
                sampled_pixels += 1
                rgb = self._frame_rgb_at(frame, x, y)
                if rgb is None:
                    continue
                object_id = color_map.get(rgb)
                if not object_id:
                    continue
                mapped_pixels += 1
                object_id = str(object_id)
                if object_id in held_ids:
                    held_pixels += 1
                    continue
                counts[object_id] = counts.get(object_id, 0) + 1
                if object_id == target_id:
                    distance = (float(x) - preferred_x) ** 2 + (float(y) - preferred_y) ** 2
                    if distance < best_distance:
                        best_distance = distance
                        best_point = (x, y)

        counted_pixels = sum(counts.values())
        target_pixels = int(counts.get(target_id, 0))
        target_ratio = target_pixels / counted_pixels if counted_pixels > 0 else 0.0
        selected_point = None
        if best_point is not None:
            selected_point = (
                min(0.95, max(0.05, best_point[0] / max(1.0, float(width)))),
                min(0.95, max(0.05, best_point[1] / max(1.0, float(height)))),
            )

        return {
            "segmentation_available": True,
            "sampled_pixels": sampled_pixels,
            "mapped_pixels": mapped_pixels,
            "counted_pixels": counted_pixels,
            "held_object_pixels": held_pixels,
            "target_instance_pixels": target_pixels,
            "target_instance_ratio": round(target_ratio, 4),
            "selected_point_norm": selected_point,
        }

    def _validate_visual_receptacle_grounding(self, visual_candidate: JsonDict, receptacle: JsonDict) -> JsonDict:
        stats = self._visual_bbox_stats(visual_candidate)
        explicit_surface_point = self._is_explicit_surface_place_candidate(visual_candidate)
        place_point = self._visual_place_point(visual_candidate, receptacle)
        if place_point is None:
            return {
                "passed": False,
                "result_type": "error_visual_receptacle_no_interaction_point",
                "message": "No visual placement point could be derived from the receptacle candidate.",
                **stats,
            }

        bbox_evidence = self._instance_bbox_evidence(
            target_object_id=receptacle.get("objectId"),
            visual_candidate=visual_candidate,
        )
        if (
            not explicit_surface_point
            and stats.get("broad_front_receptacle")
            and bbox_evidence.get("segmentation_available")
        ):
            ratio_env_name = (
                "ROBOT_PLACE_FRONT_EDGE_BBOX_MIN_RATIO"
                if stats.get("front_edge_receptacle")
                else "ROBOT_PLACE_VISUAL_BBOX_MIN_RATIO"
            )
            bbox_min_ratio = _env_float(ratio_env_name, 0.01 if stats.get("front_edge_receptacle") else 0.03)
            bbox_min_pixels = max(1, int(_env_float("ROBOT_PLACE_VISUAL_BBOX_MIN_PIXELS", 12.0)))
            bbox_target_ratio = float(bbox_evidence.get("target_instance_ratio", 0.0) or 0.0)
            bbox_target_pixels = int(bbox_evidence.get("target_instance_pixels", 0) or 0)
            selected_point = bbox_evidence.get("selected_point_norm")
            if (
                bbox_target_pixels >= bbox_min_pixels
                and bbox_target_ratio >= bbox_min_ratio
                and isinstance(selected_point, tuple)
            ):
                return {
                    "passed": True,
                    "result_type": "visual_receptacle_bbox_instance_grounded",
                    "message": "The broad front receptacle box overlaps the selected AI2-THOR receptacle instance.",
                    "interaction_point": {"x": round(selected_point[0], 3), "y": round(selected_point[1], 3)},
                    "min_target_instance_ratio": bbox_min_ratio,
                    "min_target_instance_pixels": bbox_min_pixels,
                    "evidence_source": "bbox_instance_overlap",
                    **stats,
                    **bbox_evidence,
                }
            return {
                "passed": False,
                "result_type": "error_visual_receptacle_instance_mismatch",
                "message": "The broad front receptacle box does not overlap the selected AI2-THOR receptacle instance enough.",
                "interaction_point": {"x": round(place_point[0], 3), "y": round(place_point[1], 3)},
                "min_target_instance_ratio": bbox_min_ratio,
                "min_target_instance_pixels": bbox_min_pixels,
                "evidence_source": "bbox_instance_overlap",
                **stats,
                **bbox_evidence,
            }

        evidence = self._instance_patch_evidence(
            target_object_id=receptacle.get("objectId"),
            point_norm=place_point,
        )
        min_ratio = _env_float("ROBOT_PLACE_VISUAL_TARGET_MIN_RATIO", 0.25)
        min_pixels = max(1, int(_env_float("ROBOT_PLACE_VISUAL_TARGET_MIN_PIXELS", 4.0)))
        ambiguous_min_ratio = _env_float("ROBOT_PLACE_VISUAL_AMBIGUOUS_MIN_RATIO", 0.55)
        target_ratio = float(evidence.get("target_instance_ratio", 0.0) or 0.0)
        target_pixels = int(evidence.get("target_instance_pixels", 0) or 0)

        if evidence.get("segmentation_available"):
            if target_pixels < min_pixels or target_ratio < min_ratio:
                return {
                    "passed": False,
                    "result_type": "error_visual_receptacle_instance_mismatch",
                    "message": "The visual placement point is not grounded on the selected receptacle instance.",
                    "interaction_point": {"x": round(place_point[0], 3), "y": round(place_point[1], 3)},
                    "min_target_instance_ratio": min_ratio,
                    "min_target_instance_pixels": min_pixels,
                    **stats,
                    **evidence,
                }
            if (
                not explicit_surface_point
                and stats.get("box_ambiguous")
                and target_ratio < ambiguous_min_ratio
            ):
                return {
                    "passed": False,
                    "result_type": "error_visual_receptacle_ambiguous",
                    "message": "The visual receptacle box is too broad and lacks strong target-instance evidence.",
                    "interaction_point": {"x": round(place_point[0], 3), "y": round(place_point[1], 3)},
                    "min_target_instance_ratio": ambiguous_min_ratio,
                    **stats,
                    **evidence,
                }
            return {
                "passed": True,
                "result_type": "visual_receptacle_instance_grounded",
                "message": "The visual placement point is grounded on the selected receptacle instance.",
                "interaction_point": {"x": round(place_point[0], 3), "y": round(place_point[1], 3)},
                **stats,
                **evidence,
            }

        if stats.get("box_ambiguous") and not explicit_surface_point:
            return {
                "passed": False,
                "result_type": "error_visual_receptacle_ambiguous",
                "message": "The visual receptacle box is too broad and no instance segmentation evidence is available.",
                "interaction_point": {"x": round(place_point[0], 3), "y": round(place_point[1], 3)},
                **stats,
                **evidence,
            }

        detection_bbox = self._detection_bbox_for_object_id(receptacle.get("objectId"))
        pixel_point = (place_point[0] * float(self.width), place_point[1] * float(self.height))
        if detection_bbox is not None and self._point_in_bbox(pixel_point, detection_bbox):
            return {
                "passed": True,
                "result_type": "visual_receptacle_bbox_grounded",
                "message": "The visual placement point is inside the selected receptacle 2D detection.",
                "interaction_point": {"x": round(place_point[0], 3), "y": round(place_point[1], 3)},
                **stats,
                **evidence,
            }
        return {
            "passed": False,
            "result_type": "error_visual_receptacle_not_grounded",
            "message": "The visual receptacle candidate is not grounded by instance or 2D detection evidence.",
            "interaction_point": {"x": round(place_point[0], 3), "y": round(place_point[1], 3)},
            **stats,
            **evidence,
        }

    def _allow_front_edge_interactable_grounding_override(
        self,
        *,
        visual_candidate: JsonDict,
        receptacle: JsonDict,
        grounding: JsonDict,
        interactable_guidance: JsonDict,
    ) -> JsonDict:
        """Bridge YOLO front-edge boxes with AI2-THOR interactable-pose grounding.

        CounterTop front edges often occupy a visible, reachable screen strip
        whose instance mask does not overlap the selected tabletop object well.
        ALFRED-style execution should trust the simulator's interactable pose
        check more than that mask failure, but only for front-edge receptacles.
        """
        if not _env_bool("ROBOT_PLACE_FRONT_EDGE_INTERACTABLE_OVERRIDE", True):
            return {"passed": False, "reason": "front_edge_interactable_override_disabled"}
        if self._is_explicit_surface_place_candidate(visual_candidate):
            return {"passed": False, "reason": "surface_point_requires_point_grounding"}

        result_type = str(grounding.get("result_type") or "")
        if result_type not in {
            "error_visual_receptacle_instance_mismatch",
            "error_visual_receptacle_not_grounded",
        }:
            return {"passed": False, "reason": "non_overridable_grounding_failure"}

        stats = self._visual_bbox_stats(visual_candidate)
        if not bool(stats.get("front_edge_receptacle") or stats.get("broad_front_receptacle")):
            return {"passed": False, "reason": "not_front_edge_receptacle"}
        if not bool(receptacle.get("allowed_place_receptacle") and receptacle.get("receptacle")):
            return {"passed": False, "reason": "not_actionable_receptacle"}
        if not bool(interactable_guidance.get("available", False)):
            return {"passed": False, "reason": "interactable_pose_unavailable"}

        if not bool(interactable_guidance.get("current_pose_interactable", False)):
            action = str(interactable_guidance.get("recommended_action") or "")
            return {
                "passed": False,
                "needs_approach": action in {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight", "RotateLeft", "RotateRight"},
                "reason": "front_edge_receptacle_needs_interactable_pose",
            }

        return {
            "passed": True,
            "reason": "front_edge_receptacle_current_pose_interactable",
        }

    def _is_explicit_surface_place_candidate(self, visual_candidate: JsonDict) -> bool:
        """Return whether placement is already expressed as a checked surface point."""
        if str(visual_candidate.get("surface_candidate_source") or "") not in EXPLICIT_SURFACE_PLACE_SOURCES:
            return False
        if not bool(visual_candidate.get("visual_place_ready")):
            return False
        if not bool(visual_candidate.get("affordance_ready")):
            return False
        if visual_candidate.get("rejection_reasons"):
            return False
        if bool(visual_candidate.get("blocked")) or not bool(visual_candidate.get("reachable")):
            return False
        point = visual_candidate.get("interaction_point")
        if not isinstance(point, dict):
            return False
        try:
            x = float(point.get("x"))
            y = float(point.get("y"))
        except (TypeError, ValueError):
            return False
        return math.isfinite(x) and math.isfinite(y)

    def _apply_grounded_surface_place_rule(
        self,
        *,
        visual_candidate: JsonDict,
        receptacle: JsonDict,
        grounding: JsonDict,
    ) -> None:
        """Use a grounded free-space point instead of the receptacle centroid.

        Large countertops can have an off-center metadata centroid while a
        locally free, reachable screen point is valid. The subsequent
        interactable-pose and visual-point checks remain authoritative.
        """
        if not self._is_explicit_surface_place_candidate(visual_candidate):
            return
        if str(grounding.get("result_type") or "") not in {
            "visual_receptacle_instance_grounded",
            "visual_receptacle_bbox_grounded",
            "visual_receptacle_bbox_instance_grounded",
        }:
            return
        receptacle["visual_surface_interaction_point_ready"] = True
        receptacle["visual_place_rule_source"] = "grounded_surface_interaction_point"
        receptacle["place_rule_passed"] = bool(
            receptacle.get("allowed_place_receptacle")
            and receptacle.get("receptacle")
        )

    def _ground_visual_service_candidate(
        self,
        candidates: List[JsonDict],
        *,
        visual_candidate: JsonDict,
        task_semantic_class: str,
    ) -> Optional[JsonDict]:
        if not visual_candidate:
            return None
        if visual_candidate.get("task_semantic_class") not in {None, "", task_semantic_class}:
            return None

        visual_labels = self._visual_label_tokens(visual_candidate)
        visual_bbox = self._visual_bbox(visual_candidate)
        visual_point = self._visual_point(visual_candidate)
        visual_position_hint = str(visual_candidate.get("position_hint") or "")
        visual_stats = self._visual_bbox_stats(visual_candidate)
        broad_front_receptacle = bool(
            task_semantic_class == "place_receptacle"
            and visual_stats.get("broad_front_receptacle")
            and not self._is_explicit_surface_place_candidate(visual_candidate)
        )

        scored: List[Tuple[float, JsonDict]] = []
        for candidate in candidates:
            object_token = self._label_token(candidate.get("objectType"))
            label_match = bool(visual_labels and object_token in visual_labels)
            detection_bbox = self._detection_bbox_for_object_id(candidate.get("objectId"))
            bbox_evidence: JsonDict = {}
            score = 0.0
            if label_match:
                score += 20.0
            if detection_bbox is not None and visual_bbox is not None:
                score += self._bbox_iou(detection_bbox, visual_bbox) * 12.0
            if detection_bbox is not None and visual_point is not None:
                if self._point_in_bbox(visual_point, detection_bbox):
                    score += 8.0
                else:
                    dx = ((detection_bbox[0] + detection_bbox[2]) / 2.0) - visual_point[0]
                    dy = ((detection_bbox[1] + detection_bbox[3]) / 2.0) - visual_point[1]
                    distance = math.hypot(dx, dy)
                    score += max(0.0, 6.0 - distance / 40.0)
            if visual_position_hint and candidate.get("position_hint") == visual_position_hint:
                score += 2.0
            if candidate.get("is_front_center"):
                score += 1.0
            if broad_front_receptacle:
                if candidate.get("allowed_place_receptacle") and candidate.get("receptacle"):
                    score += 4.0
                if candidate.get("is_near_place"):
                    score += 6.0
                bbox_evidence = self._instance_bbox_evidence(
                    target_object_id=candidate.get("objectId"),
                    visual_candidate=visual_candidate,
                )
                target_pixels = int(bbox_evidence.get("target_instance_pixels", 0) or 0)
                target_ratio = float(bbox_evidence.get("target_instance_ratio", 0.0) or 0.0)
                if target_pixels > 0:
                    score += min(8.0, math.log1p(float(target_pixels)) * 2.0)
                    score += min(4.0, target_ratio * 80.0)
                if isinstance(bbox_evidence.get("selected_point_norm"), tuple):
                    score += 4.0
            try:
                score += max(0.0, 1.5 - float(candidate.get("ground_distance", 9.0)))
            except (TypeError, ValueError):
                pass
            if score > 0.0:
                grounded = dict(candidate)
                grounded["visual_grounding_score"] = round(score, 3)
                grounded["visual_grounding_label_match"] = bool(label_match)
                grounded["visual_grounding_has_2d_detection"] = detection_bbox is not None
                if bbox_evidence:
                    grounded["visual_bbox_instance_pixels"] = bbox_evidence.get("target_instance_pixels", 0)
                    grounded["visual_bbox_instance_ratio"] = bbox_evidence.get("target_instance_ratio", 0.0)
                    grounded["visual_bbox_has_selected_point"] = bool(
                        isinstance(bbox_evidence.get("selected_point_norm"), tuple)
                    )
                scored.append((score, grounded))

        if not scored:
            return None
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best = scored[0]
        if visual_labels and not best.get("visual_grounding_label_match") and best_score < 6.0:
            return None
        return best

    def _get_visible_pickup_candidates(self) -> List[JsonDict]:
        candidates: List[JsonDict] = []
        for obj in self.last_event.metadata.get("objects", []):
            if obj.get("objectType") not in self.pickup_target_types:
                continue
            if not obj.get("visible", False):
                continue
            candidates.append(self._summarize_service_object(obj, task_semantic_class="pickup_target"))
        candidates.sort(key=lambda c: c.get("ground_distance", float("inf")))
        return candidates

    def _get_visible_receptacle_candidates(self) -> List[JsonDict]:
        candidates: List[JsonDict] = []
        for obj in self.last_event.metadata.get("objects", []):
            if not obj.get("visible", False):
                continue
            if obj.get("objectType") not in self.place_receptacle_types:
                continue
            candidates.append(self._summarize_service_object(obj, task_semantic_class="place_receptacle"))
        candidates.sort(key=lambda c: c.get("ground_distance", float("inf")))
        return candidates

    def _summarize_service_object(self, obj: JsonDict, *, task_semantic_class: str) -> JsonDict:
        summary = self._summarize_object(obj)
        position_hint = self._relative_position_hint(obj)
        ground_distance = self._ground_distance_to_object(obj)
        is_front_center = position_hint == "front-center"
        is_near_pick = ground_distance <= float(os.getenv("ROBOT_PICK_MAX_DISTANCE", "1.0"))
        is_near_place = ground_distance <= float(os.getenv("ROBOT_PLACE_MAX_DISTANCE", "1.0"))
        is_pickupable = bool(obj.get("pickupable", False)) or obj.get("objectType") in self.pickup_target_types
        is_allowed_place_receptacle = obj.get("objectType") in self.place_receptacle_types
        is_receptacle = bool(obj.get("receptacle", False)) or is_allowed_place_receptacle
        summary.update(
            {
                "task_semantic_class": task_semantic_class,
                "position_hint": position_hint,
                "angle_delta_deg": round(self._relative_angle_delta(obj), 2),
                "ground_distance": round(ground_distance, 3),
                "is_front_center": bool(is_front_center),
                "is_near_pick": bool(is_near_pick),
                "is_near_place": bool(is_near_place),
                "pickupable": bool(is_pickupable),
                "receptacle": bool(is_receptacle),
                "allowed_place_receptacle": bool(is_allowed_place_receptacle),
                "pickup_rule_passed": bool(task_semantic_class == "pickup_target" and is_pickupable and is_front_center and is_near_pick),
                "place_rule_passed": bool(
                    task_semantic_class == "place_receptacle"
                    and is_allowed_place_receptacle
                    and is_receptacle
                    and is_front_center
                    and is_near_place
                ),
            }
        )
        return summary

    def _infer_pick_reject_type(self, candidates: List[JsonDict]) -> str:
        if not candidates:
            return "error_no_pickup_target"
        first = candidates[0]
        if not first.get("is_front_center"):
            return "error_pickup_target_not_centered"
        if not first.get("is_near_pick"):
            return "error_pickup_target_too_far"
        if not first.get("pickupable"):
            return "error_target_not_pickupable"
        return "error_no_pickup_target"

    def _infer_place_reject_type(self, candidates: List[JsonDict]) -> str:
        if not candidates:
            return "error_no_receptacle"
        first = candidates[0]
        if not first.get("is_front_center"):
            return "error_receptacle_not_centered"
        if not first.get("is_near_place"):
            return "error_receptacle_too_far"
        if not first.get("receptacle"):
            return "error_target_not_receptacle"
        return "error_no_receptacle"

    # ------------------------------------------------------------------
    # Geometry/object helpers
    # ------------------------------------------------------------------

    def _select_seed_object(self, object_types: Sequence[str]) -> Optional[JsonDict]:
        requested = [item for item in object_types if item]
        objects = self.last_event.metadata.get("objects", [])
        for preferred_type in requested:
            for obj in objects:
                if obj.get("objectType") != preferred_type:
                    continue
                obj_id = str(obj.get("objectId"))
                if obj_id in self._used_seed_object_ids:
                    continue
                return obj
        # Fallback: allow reusing an already used object if no other proxy exists.
        for preferred_type in requested:
            for obj in objects:
                if obj.get("objectType") == preferred_type:
                    return obj
        return None

    def _target_position_from_layout(
        self,
        *,
        layout: str,
        distance: float,
        y: Optional[float],
        lateral: Optional[float],
        kind: str,
    ) -> JsonDict:
        agent = self.last_event.metadata["agent"]
        agent_pos = agent["position"]
        agent_rot_y = float(agent["rotation"]["y"])
        rad = math.radians(agent_rot_y)
        forward_x, forward_z = math.sin(rad), math.cos(rad)
        right_x, right_z = math.cos(rad), -math.sin(rad)

        layout_defaults = {
            "front-center": (distance, 0.0),
            "front-left": (distance, -0.38),
            "front-right": (distance, 0.38),
            "near-front": (0.45, 0.0),
            "far-front": (1.05, 0.0),
            "left-near": (0.55, -0.55),
            "right-near": (0.55, 0.55),
        }
        forward_dist, lateral_offset = layout_defaults.get(layout, (distance, 0.0))
        if lateral is not None:
            lateral_offset = float(lateral)

        target_x = float(agent_pos["x"]) + forward_dist * forward_x + lateral_offset * right_x
        target_z = float(agent_pos["z"]) + forward_dist * forward_z + lateral_offset * right_z
        if y is None:
            if kind in {"floor_trash", "pickup_target", "obstacle"}:
                y = 0.1
            elif kind == "place_receptacle":
                y = 0.72
            else:
                y = 0.85
        return {"x": round(target_x, 4), "y": float(y), "z": round(target_z, 4)}

    def _task_class_for_seed_kind(self, kind: str) -> str:
        mapping = {
            "floor_trash": "cleanable_floor_trash",
            "non_floor_decoy": "non_floor_object",
            "pickup_target": "pickup_target",
            "place_receptacle": "place_receptacle",
            "obstacle": "obstacle",
        }
        return mapping.get(str(kind or ""), "ignored_object")

    def _summarize_object(self, obj: JsonDict) -> JsonDict:
        return {
            "objectId": obj.get("objectId"),
            "objectType": obj.get("objectType"),
            "visible": bool(obj.get("visible", False)),
            "distance": float(obj.get("distance", float("inf"))),
            "position": obj.get("position", {}),
        }

    def _summarize_annotation_object(self, obj: JsonDict) -> JsonDict:
        return {
            "objectId": obj.get("objectId"),
            "objectType": obj.get("objectType"),
            "visible": bool(obj.get("visible", False)),
            "pickupable": bool(obj.get("pickupable", False)),
            "receptacle": bool(obj.get("receptacle", False)),
        }

    def _instance_detections_2d(self) -> JsonDict:
        converted = self._converted_instance_detections_2d()
        if converted:
            return converted
        converted = self._instance_detections_from_masks()
        if converted:
            return converted
        return self._instance_detections_from_segmentation_frame()

    def _converted_instance_detections_2d(self) -> JsonDict:
        detections = getattr(self.last_event, "instance_detections2D", None)
        if not isinstance(detections, dict):
            detections = self.last_event.metadata.get("instanceDetections2D", {})
        if not isinstance(detections, dict):
            return {}
        converted: JsonDict = {}
        for object_id, box in detections.items():
            try:
                values = box.tolist() if hasattr(box, "tolist") else list(box)
                if len(values) >= 4:
                    converted[str(object_id)] = [float(v) for v in values[:4]]
            except Exception:
                continue
        return converted

    def _instance_detections_from_masks(self) -> JsonDict:
        masks = getattr(self.last_event, "instance_masks", None)
        if not isinstance(masks, dict):
            masks = self.last_event.metadata.get("instanceMasks", {})
        if not isinstance(masks, dict):
            return {}

        converted: JsonDict = {}
        for object_id, mask in masks.items():
            box = self._bbox_from_mask(mask)
            if box is not None:
                converted[str(object_id)] = box
        return converted

    def _bbox_from_mask(self, mask: Any) -> Optional[List[float]]:
        try:
            if hasattr(mask, "nonzero"):
                ys, xs = mask.nonzero()
                if len(xs) == 0 or len(ys) == 0:
                    return None
                return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]

            min_x: Optional[int] = None
            min_y: Optional[int] = None
            max_x: Optional[int] = None
            max_y: Optional[int] = None
            for y, row in enumerate(mask):
                for x, value in enumerate(row):
                    if not value:
                        continue
                    min_x = x if min_x is None else min(min_x, x)
                    min_y = y if min_y is None else min(min_y, y)
                    max_x = x if max_x is None else max(max_x, x)
                    max_y = y if max_y is None else max(max_y, y)
            if min_x is None or min_y is None or max_x is None or max_y is None:
                return None
            return [float(min_x), float(min_y), float(max_x + 1), float(max_y + 1)]
        except Exception:
            return None

    def _instance_detections_from_segmentation_frame(self) -> JsonDict:
        frame = getattr(self.last_event, "instance_segmentation_frame", None)
        color_map = self._instance_color_to_object_id()
        if frame is None or not color_map:
            return {}

        converted: JsonDict = {}
        try:
            pixels = frame[:, :, :3]
        except Exception:
            return self._instance_detections_from_nested_segmentation_frame(frame, color_map)

        for rgb, object_id in color_map.items():
            try:
                r, g, b = rgb
                mask = (pixels[:, :, 0] == r) & (pixels[:, :, 1] == g) & (pixels[:, :, 2] == b)
                ys, xs = mask.nonzero()
                if len(xs) == 0 or len(ys) == 0:
                    continue
                converted[str(object_id)] = [
                    float(xs.min()),
                    float(ys.min()),
                    float(xs.max() + 1),
                    float(ys.max() + 1),
                ]
            except Exception:
                continue
        return converted

    def _instance_detections_from_nested_segmentation_frame(
        self,
        frame: Any,
        color_map: Dict[Tuple[int, int, int], str],
    ) -> JsonDict:
        boxes: Dict[str, List[int]] = {}
        try:
            for y, row in enumerate(frame):
                for x, pixel in enumerate(row):
                    rgb = self._normalize_rgb_color(pixel)
                    if rgb is None or rgb not in color_map:
                        continue
                    object_id = color_map[rgb]
                    box = boxes.setdefault(str(object_id), [x, y, x, y])
                    box[0] = min(box[0], x)
                    box[1] = min(box[1], y)
                    box[2] = max(box[2], x)
                    box[3] = max(box[3], y)
        except Exception:
            return {}
        return {
            object_id: [float(x1), float(y1), float(x2 + 1), float(y2 + 1)]
            for object_id, (x1, y1, x2, y2) in boxes.items()
        }

    def _instance_color_to_object_id(self) -> Dict[Tuple[int, int, int], str]:
        color_map: Dict[Tuple[int, int, int], str] = {}

        color_to_object_id = getattr(self.last_event, "color_to_object_id", None)
        if isinstance(color_to_object_id, dict):
            for raw_color, object_id in color_to_object_id.items():
                rgb = self._normalize_rgb_color(raw_color)
                if rgb is not None and object_id:
                    color_map[rgb] = str(object_id)

        object_id_to_color = getattr(self.last_event, "object_id_to_color", None)
        if isinstance(object_id_to_color, dict):
            for object_id, raw_color in object_id_to_color.items():
                rgb = self._normalize_rgb_color(raw_color)
                if rgb is not None and object_id:
                    color_map[rgb] = str(object_id)

        metadata_color_to_object_id = self.last_event.metadata.get("colorToObjectId", {})
        if isinstance(metadata_color_to_object_id, dict):
            for raw_color, object_id in metadata_color_to_object_id.items():
                rgb = self._normalize_rgb_color(raw_color)
                if rgb is not None and object_id:
                    color_map[rgb] = str(object_id)

        metadata_object_id_to_color = self.last_event.metadata.get("objectIdToColor", {})
        if isinstance(metadata_object_id_to_color, dict):
            for object_id, raw_color in metadata_object_id_to_color.items():
                rgb = self._normalize_rgb_color(raw_color)
                if rgb is not None and object_id:
                    color_map[rgb] = str(object_id)

        return color_map

    def _normalize_rgb_color(self, value: Any) -> Optional[Tuple[int, int, int]]:
        try:
            if hasattr(value, "tolist"):
                value = value.tolist()
            if isinstance(value, dict):
                if {"r", "g", "b"}.issubset(value.keys()):
                    parts = [value["r"], value["g"], value["b"]]
                elif {"red", "green", "blue"}.issubset(value.keys()):
                    parts = [value["red"], value["green"], value["blue"]]
                else:
                    return None
            elif isinstance(value, str):
                text = value.strip()
                if text.startswith("#") and len(text) >= 7:
                    return (int(text[1:3], 16), int(text[3:5], 16), int(text[5:7], 16))
                for char in "(),[]":
                    text = text.replace(char, " ")
                parts = [item for item in text.replace(",", " ").split() if item]
            else:
                parts = list(value)
            if len(parts) < 3:
                return None
            rgb = tuple(int(round(float(part))) for part in parts[:3])
            if any(channel < 0 or channel > 255 for channel in rgb):
                return None
            return rgb  # type: ignore[return-value]
        except Exception:
            return None

    def _segmentation_debug(self) -> JsonDict:
        detections = getattr(self.last_event, "instance_detections2D", None)
        if not isinstance(detections, dict):
            detections = self.last_event.metadata.get("instanceDetections2D", {})
        masks = getattr(self.last_event, "instance_masks", None)
        if not isinstance(masks, dict):
            masks = self.last_event.metadata.get("instanceMasks", {})
        segmentation_frame = getattr(self.last_event, "instance_segmentation_frame", None)
        color_map = self._instance_color_to_object_id()
        return {
            "instance_detections2D_count": len(detections) if isinstance(detections, dict) else 0,
            "instance_masks_count": len(masks) if isinstance(masks, dict) else 0,
            "bbox_fallback_count": len(self._instance_detections_from_masks()),
            "segmentation_color_map_count": len(color_map),
            "segmentation_frame_bbox_count": len(self._instance_detections_from_segmentation_frame()),
            "has_instance_segmentation_frame": segmentation_frame is not None,
        }

    def _ground_distance_to_object(self, obj: JsonDict) -> float:
        agent = self.last_event.metadata["agent"]
        agent_pos = agent["position"]
        obj_pos = obj.get("position", {})
        dx = float(obj_pos.get("x", 0.0)) - float(agent_pos.get("x", 0.0))
        dz = float(obj_pos.get("z", 0.0)) - float(agent_pos.get("z", 0.0))
        return math.sqrt(dx * dx + dz * dz)

    def _find_object_by_id(self, object_id: str) -> Optional[JsonDict]:
        for obj in self.last_event.metadata.get("objects", []):
            if obj.get("objectId") == object_id:
                return obj
        return None

    def _id_in_list(self, object_id: Any, values: Any) -> bool:
        if not object_id or not isinstance(values, list):
            return False
        target = str(object_id)
        return any(str(item) == target for item in values)

    def _object_in_receptacle(self, obj: JsonDict, receptacle: JsonDict) -> bool:
        object_id = obj.get("objectId")
        receptacle_id = receptacle.get("objectId")
        if self._id_in_list(receptacle_id, obj.get("parentReceptacles")):
            return True
        if self._id_in_list(object_id, receptacle.get("receptacleObjectIds")):
            return True
        return False

    def _distance_bucket(self, distance: Optional[float]) -> str:
        if distance is None:
            return "unknown"
        if distance <= 0.75:
            return "near"
        if distance <= 1.0:
            return "reachable"
        if distance <= 1.25:
            return "borderline"
        return "far"

    def _angle_bucket(self, angle_abs: Optional[float]) -> str:
        if angle_abs is None:
            return "unknown"
        if angle_abs <= 15:
            return "front-center"
        if angle_abs <= 35:
            return "front-reachable"
        if angle_abs <= 60:
            return "side"
        return "outside-front"

    def _has_grid_clearance_contract(self, visual_candidate: JsonDict) -> bool:
        """Return whether point clearance was fully evaluated in the RGB-D grid."""
        if str(visual_candidate.get("surface_candidate_source") or "") != "pointcloud_plane_grid_completion":
            return False
        if not self._is_explicit_surface_place_candidate(visual_candidate):
            return False
        contract = (
            visual_candidate.get("placement_safety_contract")
            if isinstance(visual_candidate.get("placement_safety_contract"), dict)
            else {}
        )
        completion = (
            visual_candidate.get("free_space_completion")
            if isinstance(visual_candidate.get("free_space_completion"), dict)
            else {}
        )
        checks = visual_candidate.get("geometry_checks") if isinstance(visual_candidate.get("geometry_checks"), dict) else {}
        occupancy = (
            visual_candidate.get("occupancy_checks")
            if isinstance(visual_candidate.get("occupancy_checks"), dict)
            else {}
        )
        return bool(
            contract.get("version") == "plane_local_grid_v1"
            and contract.get("clearance_owner") == "pointcloud_plane_local_grid"
            and contract.get("grid_occupancy_clear")
            and contract.get("grid_edge_eroded")
            and completion.get("mode") == "plane_local_2d_grid"
            and checks.get("grid_occupancy_clear")
            and checks.get("grid_edge_eroded")
            and checks.get("distance_ok")
            and occupancy.get("free_space_grid_completion")
            and not occupancy.get("blocked")
        )

    def _grid_exact_target_required(self, visual_candidate: JsonDict) -> bool:
        """Return whether a metric free-space point must be executed as a world target."""
        return self._has_grid_clearance_contract(visual_candidate)

    def _camera_relative_surface_target_to_world(
        self,
        center_3d: JsonDict,
        visual_candidate: JsonDict,
    ) -> Optional[JsonDict]:
        """Transform an RGB-D surface target into AI2-THOR world coordinates.

        Perception reports x as camera-right, y as height above the observed
        floor and z as ground-forward. The actuator uses the current agent
        pose only to express the perception-selected point in world space.
        """
        contract = (
            visual_candidate.get("placement_safety_contract")
            if isinstance(visual_candidate.get("placement_safety_contract"), dict)
            else {}
        )
        frame = str(
            contract.get("target_coordinate_frame")
            or "camera_relative_x_right_y_height_z_forward"
        )
        if frame != "camera_relative_x_right_y_height_z_forward":
            return None
        agent = self.last_event.metadata.get("agent") if isinstance(self.last_event.metadata, dict) else {}
        position = agent.get("position") if isinstance(agent, dict) and isinstance(agent.get("position"), dict) else {}
        rotation = agent.get("rotation") if isinstance(agent, dict) and isinstance(agent.get("rotation"), dict) else {}
        try:
            lateral = float(center_3d.get("x"))
            height_from_floor = float(center_3d.get("y"))
            forward = float(center_3d.get("ground_forward_m", center_3d.get("z")))
            agent_x = float(position.get("x"))
            agent_y = float(position.get("y"))
            agent_z = float(position.get("z"))
            agent_rotation_y = float(rotation.get("y", 0.0))
            camera_height_m = float(
                contract.get(
                    "camera_height_m",
                    _env_float("ROBOT_CAMERA_HEIGHT_M", agent_y),
                )
            )
        except (TypeError, ValueError):
            return None
        values = (
            lateral,
            height_from_floor,
            forward,
            agent_x,
            agent_y,
            agent_z,
            agent_rotation_y,
            camera_height_m,
        )
        if not all(math.isfinite(value) for value in values):
            return None
        radians = math.radians(agent_rotation_y)
        forward_x, forward_z = math.sin(radians), math.cos(radians)
        right_x, right_z = math.cos(radians), -math.sin(radians)
        floor_world_y = agent_y - camera_height_m
        return {
            "x": round(agent_x + forward * forward_x + lateral * right_x, 4),
            "y": round(floor_world_y + height_from_floor, 4),
            "z": round(agent_z + forward * forward_z + lateral * right_z, 4),
        }

    def _grid_contract_world_targets(self, visual_candidate: JsonDict) -> List[JsonDict]:
        """Return ranked perception-selected world targets for exact execution."""
        if not self._grid_exact_target_required(visual_candidate):
            return []
        raw_points = visual_candidate.get("placement_points")
        items = raw_points if isinstance(raw_points, list) else [visual_candidate.get("interaction_point")]
        targets: List[JsonDict] = []
        for index, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            center_3d = item.get("center_3d") if isinstance(item.get("center_3d"), dict) else None
            if center_3d is None:
                continue
            world_target = self._camera_relative_surface_target_to_world(center_3d, visual_candidate)
            if world_target is None:
                continue
            try:
                rank = int(item.get("rank", index) or index)
            except (TypeError, ValueError):
                rank = index
            targets.append({"rank": rank, "surface_world_target": world_target})
        targets.sort(key=lambda item: int(item.get("rank", 0) or 0))
        return targets

    def _resolve_grid_exact_targets(
        self,
        visual_candidate: JsonDict,
        receptacle: JsonDict,
    ) -> Tuple[List[JsonDict], JsonDict]:
        """Snap RGB-D targets to simulator-legal positions near the same surface cell."""
        requested = self._grid_contract_world_targets(visual_candidate)
        detail: JsonDict = {
            "requested_target_count": len(requested),
            "resolved_target_count": 0,
            "mode": "world_point_exact",
        }
        if not requested:
            detail["reason"] = "missing_metric_grid_targets"
            return [], detail
        receptacle_id = receptacle.get("objectId")
        if not receptacle_id:
            detail["reason"] = "missing_grounded_receptacle_id"
            return [], detail
        try:
            event = self.controller.step(
                action="GetSpawnCoordinatesAboveReceptacle",
                objectId=str(receptacle_id),
                anywhere=True,
            )
        except Exception as exc:
            detail["reason"] = "spawn_coordinate_query_exception"
            detail["error_message"] = str(exc)
            return [], detail
        self.last_event = event
        metadata = event.metadata if hasattr(event, "metadata") and isinstance(event.metadata, dict) else {}
        if not bool(metadata.get("lastActionSuccess", False)):
            detail["reason"] = "spawn_coordinate_query_failed"
            detail["error_message"] = metadata.get("errorMessage", "")
            return [], detail
        coordinates = metadata.get("actionReturn") or []
        coordinates = coordinates if isinstance(coordinates, list) else []
        legal_positions: List[JsonDict] = []
        for position in coordinates:
            if not isinstance(position, dict):
                continue
            try:
                values = {axis: float(position.get(axis)) for axis in ("x", "y", "z")}
            except (TypeError, ValueError):
                continue
            if all(math.isfinite(value) for value in values.values()):
                legal_positions.append(values)
        detail["legal_target_count"] = len(legal_positions)
        if not legal_positions:
            detail["reason"] = "no_legal_spawn_coordinates"
            return [], detail

        max_error = max(0.0, _env_float("ROBOT_PLACE_EXACT_RESOLVE_MAX_ERROR_M", 0.10))
        resolved: List[JsonDict] = []
        used: Set[Tuple[int, int, int]] = set()
        for item in requested:
            requested_position = item["surface_world_target"]
            nearest = min(
                legal_positions,
                key=lambda position: math.hypot(
                    float(position["x"]) - float(requested_position["x"]),
                    float(position["z"]) - float(requested_position["z"]),
                ),
            )
            error_m = math.hypot(
                float(nearest["x"]) - float(requested_position["x"]),
                float(nearest["z"]) - float(requested_position["z"]),
            )
            key = (
                int(round(float(nearest["x"]) * 10000)),
                int(round(float(nearest["y"]) * 10000)),
                int(round(float(nearest["z"]) * 10000)),
            )
            if error_m > max_error or key in used:
                continue
            used.add(key)
            resolved.append(
                {
                    "rank": item.get("rank"),
                    "surface_world_target": requested_position,
                    "action_world_target": {
                        axis: round(float(nearest[axis]), 4)
                        for axis in ("x", "y", "z")
                    },
                    "resolution_error_m": round(float(error_m), 4),
                }
            )
        detail["resolved_target_count"] = len(resolved)
        detail["max_resolution_error_m"] = round(float(max_error), 4)
        if resolved:
            detail["best_resolution_error_m"] = resolved[0]["resolution_error_m"]
            detail["reason"] = "exact_targets_resolved"
        else:
            detail["reason"] = "no_legal_coordinate_near_selected_grid_point"
        return resolved, detail

    def _grid_contract_place_points(self, visual_candidate: JsonDict) -> List[Tuple[float, float]]:
        if not self._has_grid_clearance_contract(visual_candidate):
            return []
        raw_points = visual_candidate.get("placement_points")
        items = raw_points if isinstance(raw_points, list) else [visual_candidate.get("interaction_point")]
        points: List[Tuple[float, float]] = []
        seen: Set[Tuple[int, int]] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                px = float(item.get("x"))
                py = float(item.get("y"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(px) or not math.isfinite(py):
                continue
            key = (int(round(px)), int(round(py)))
            if key in seen:
                continue
            seen.add(key)
            points.append((px / max(1.0, float(self.width)), py / max(1.0, float(self.height))))
        return points

    def _visual_avoidance_boxes(self, visual_candidate: JsonDict) -> List[Tuple[str, Tuple[float, float, float, float]]]:
        boxes: List[Tuple[str, Tuple[float, float, float, float]]] = []
        seen: Set[Tuple[int, int, int, int, str]] = set()
        for key in ("visible_occupants", "placement_avoidance_candidates"):
            items = visual_candidate.get(key)
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                bbox = self._visual_bbox(item)
                if bbox is None:
                    continue
                label = str(item.get("raw_label") or item.get("label") or "object")
                x1, y1, x2, y2 = bbox
                key_tuple = (
                    int(round(x1)),
                    int(round(y1)),
                    int(round(x2)),
                    int(round(y2)),
                    label.lower(),
                )
                if key_tuple in seen:
                    continue
                seen.add(key_tuple)
                boxes.append((label, bbox))
        return boxes

    def _visual_place_point_blocked(self, visual_candidate: JsonDict, x_norm: float, y_norm: float) -> bool:
        boxes = self._visual_avoidance_boxes(visual_candidate)
        if not boxes:
            return False
        px = min(0.99, max(0.0, float(x_norm))) * max(1.0, float(self.width))
        py = min(0.99, max(0.0, float(y_norm))) * max(1.0, float(self.height))
        margin_x = _env_float("ROBOT_PLACE_AVOIDANCE_MARGIN_X_PIXELS", 20.0)
        margin_y = _env_float("ROBOT_PLACE_AVOIDANCE_MARGIN_Y_PIXELS", 16.0)
        for _, bbox in boxes:
            x1, y1, x2, y2 = bbox
            if (x1 - margin_x) <= px <= (x2 + margin_x) and (y1 - margin_y) <= py <= (y2 + margin_y):
                return True
        return False

    def _visual_place_clearance_score(self, visual_candidate: JsonDict, x_norm: float, y_norm: float) -> float:
        """Return clearance from real avoidance boxes, or -1 if inside one.

        The normal point filter uses a safety margin around visible objects. If
        that margin removes every sampled point on a cluttered tabletop, use this
        stricter score as a fallback: never place inside an object's actual box,
        but allow trying the farthest gap between objects.
        """
        boxes = self._visual_avoidance_boxes(visual_candidate)
        if not boxes:
            return float("inf")
        px = min(0.99, max(0.0, float(x_norm))) * max(1.0, float(self.width))
        py = min(0.99, max(0.0, float(y_norm))) * max(1.0, float(self.height))
        best = float("inf")
        for _, bbox in boxes:
            x1, y1, x2, y2 = bbox
            if x1 <= px <= x2 and y1 <= py <= y2:
                return -1.0
            dx = max(x1 - px, 0.0, px - x2)
            dy = max(y1 - py, 0.0, py - y2)
            best = min(best, math.hypot(dx, dy))
        return best

    def _visual_place_points(self, visual_candidate: JsonDict, receptacle: JsonDict) -> List[Tuple[float, float]]:
        points: List[Tuple[float, float]] = []
        relaxed_points: List[Tuple[float, float, float]] = []
        seen: Set[Tuple[int, int]] = set()
        relaxed_seen: Set[Tuple[int, int]] = set()

        def add_point(x_norm: float, y_norm: float, *, apply_image_avoidance: bool = True) -> None:
            x_clamped = min(0.95, max(0.05, float(x_norm)))
            y_clamped = min(0.95, max(0.05, float(y_norm)))
            key = (int(round(x_clamped * 1000)), int(round(y_clamped * 1000)))
            if not apply_image_avoidance and self._visual_place_clearance_score(
                visual_candidate, x_clamped, y_clamped
            ) < 0.0:
                # A grid contract owns inflated clearance, but an execution
                # point visibly inside an occupied object is a hard
                # contradiction and must never be sent to PutObject.
                return
            if apply_image_avoidance and self._visual_place_point_blocked(visual_candidate, x_clamped, y_clamped):
                score = self._visual_place_clearance_score(visual_candidate, x_clamped, y_clamped)
                if score >= 0.0 and key not in relaxed_seen:
                    relaxed_seen.add(key)
                    relaxed_points.append((score, x_clamped, y_clamped))
                return
            if key in seen:
                return
            seen.add(key)
            points.append((x_clamped, y_clamped))

        grid_points = self._grid_contract_place_points(visual_candidate)
        if grid_points:
            for x_norm, y_norm in grid_points:
                # Pointcloud grid candidates already include the metric obstacle
                # dilation and support-edge erosion contract. Reapplying a
                # separate image-box margin here would create two conflicting
                # collision models for the same placement point. Actual-box
                # intersection is still rejected by add_point above.
                add_point(x_norm, y_norm, apply_image_avoidance=False)
            return points

        preferred_point = self._visual_place_point(visual_candidate, receptacle)
        if preferred_point is not None:
            add_point(preferred_point[0], preferred_point[1])
        if visual_candidate.get("surface_candidate_source") == "pointcloud_plane_grid_completion":
            # The bbox is the envelope of an irregular free component, not a
            # region from which additional placement points may be sampled.
            return points

        bbox = self._visual_bbox(visual_candidate)
        if bbox is None:
            bbox = self._detection_bbox_for_object_id(receptacle.get("objectId"))
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            width = max(1.0, float(self.width))
            height = max(1.0, float(self.height))
            bbox_width = max(1.0, x2 - x1)
            bbox_height = max(1.0, y2 - y1)
            stats = self._visual_bbox_stats(visual_candidate)
            broad_front = bool(stats.get("broad_front_receptacle") or stats.get("front_edge_receptacle"))
            base_y_ratio = _env_float(
                "ROBOT_PLACE_BROAD_POINT_BBOX_Y_RATIO" if broad_front else "ROBOT_PLACE_POINT_BBOX_Y_RATIO",
                0.62 if broad_front else 0.88,
            )
            if broad_front:
                y_ratios = [base_y_ratio, 0.42, 0.50, 0.58, 0.66, 0.74, 0.82]
                x_ratios = [0.50, 0.35, 0.65, 0.25, 0.75, 0.45, 0.55]
                y_min, y_max = 0.32, 0.88
            else:
                y_ratios = [base_y_ratio, 0.72, 0.82, 0.90]
                x_ratios = [0.50, 0.40, 0.60]
                y_min, y_max = 0.50, 0.95
            max_points = max(1, int(_env_float("ROBOT_PLACE_VISUAL_MAX_POINTS", 28.0 if broad_front else 12.0)))
            for y_ratio in y_ratios:
                y_ratio = min(y_max, max(y_min, float(y_ratio)))
                for x_ratio in x_ratios:
                    px = x1 + bbox_width * min(0.90, max(0.10, float(x_ratio)))
                    py = y1 + bbox_height * y_ratio
                    add_point(px / width, py / height)
                    if len(points) >= max_points:
                        return points

        visual_point = self._visual_point(visual_candidate)
        if visual_point is not None:
            width = max(1.0, float(self.width))
            height = max(1.0, float(self.height))
            add_point(visual_point[0] / width, visual_point[1] / height)
        if not points and relaxed_points:
            max_relaxed = max(1, int(_env_float("ROBOT_PLACE_RELAXED_VISUAL_MAX_POINTS", 8.0)))
            for _, x_norm, y_norm in sorted(relaxed_points, key=lambda item: item[0], reverse=True)[:max_relaxed]:
                key = (int(round(x_norm * 1000)), int(round(y_norm * 1000)))
                if key in seen:
                    continue
                seen.add(key)
                points.append((x_norm, y_norm))
        return points

    def _visual_place_point(self, visual_candidate: JsonDict, receptacle: JsonDict) -> Optional[Tuple[float, float]]:
        if visual_candidate.get("surface_candidate_source") in {
            "depth_geometry",
            "depth_region_geometry",
            "pointcloud_plane",
            "pointcloud_plane_completion",
            "pointcloud_plane_grid_completion",
        }:
            point = self._visual_point({"interaction_point": visual_candidate.get("interaction_point")})
            if point is not None:
                width = max(1.0, float(self.width))
                height = max(1.0, float(self.height))
                px, py = point
                return (
                    min(0.95, max(0.05, px / width)),
                    min(0.95, max(0.05, py / height)),
                )

        instance_evidence = self._instance_bbox_evidence(
            target_object_id=receptacle.get("objectId"),
            visual_candidate=visual_candidate,
        )
        selected_point = instance_evidence.get("selected_point_norm")
        if isinstance(selected_point, tuple) and len(selected_point) == 2:
            return (float(selected_point[0]), float(selected_point[1]))

        bbox = self._visual_bbox(visual_candidate)
        if bbox is None:
            bbox = self._detection_bbox_for_object_id(receptacle.get("objectId"))

        width = max(1.0, float(self.width))
        height = max(1.0, float(self.height))
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            stats = self._visual_bbox_stats(visual_candidate)
            y_ratio = _env_float(
                "ROBOT_PLACE_BROAD_POINT_BBOX_Y_RATIO"
                if stats.get("broad_front_receptacle")
                else "ROBOT_PLACE_POINT_BBOX_Y_RATIO",
                0.62 if stats.get("broad_front_receptacle") else 0.88,
            )
            y_ratio = min(0.95, max(0.55, y_ratio))
            px = (x1 + x2) / 2.0
            py = y1 + (y2 - y1) * y_ratio
            return (
                min(0.95, max(0.05, px / width)),
                min(0.95, max(0.05, py / height)),
            )

        point = self._visual_point(visual_candidate)
        if point is None:
            return None
        px, py = point
        return (
            min(0.95, max(0.05, px / width)),
            min(0.95, max(0.05, py / height)),
        )

    def _validate_place_result(
        self,
        *,
        held_object_id: str,
        receptacle: JsonDict,
        action_success: bool,
        inventory_empty: bool,
        expected_surface_world_target: Optional[JsonDict] = None,
        exact_target_required: bool = False,
    ) -> JsonDict:
        if not action_success:
            return {
                "placement_verified": False,
                "result_type": "error_place_failed",
                "message": "Place action was rejected by the simulator.",
                "object_on_target_receptacle": False,
                "object_reachable_from_agent": False,
                "distance_bucket": "unknown",
                "angle_bucket": "unknown",
                "placement_target_verified": False,
            }
        if not inventory_empty:
            return {
                "placement_verified": False,
                "result_type": "error_place_inventory_not_empty",
                "message": "Place action returned success but the object is still held.",
                "object_on_target_receptacle": False,
                "object_reachable_from_agent": False,
                "distance_bucket": "unknown",
                "angle_bucket": "unknown",
                "placement_target_verified": False,
            }

        placed_object = self._find_object_by_id(held_object_id)
        if placed_object is None:
            return {
                "placement_verified": False,
                "result_type": "error_place_object_missing_after_action",
                "message": "Placed object could not be found after the placement action.",
                "object_on_target_receptacle": False,
                "object_reachable_from_agent": False,
                "distance_bucket": "unknown",
                "angle_bucket": "unknown",
                "placement_target_verified": False,
            }

        receptacle_after = self._find_object_by_id(str(receptacle.get("objectId") or "")) or receptacle
        on_receptacle = self._object_in_receptacle(placed_object, receptacle_after)
        distance = self._ground_distance_to_object(placed_object)
        angle_abs = abs(self._relative_angle_delta(placed_object))
        max_distance = _env_float("ROBOT_PLACE_VERIFY_MAX_DISTANCE", _env_float("ROBOT_PICK_MAX_DISTANCE", 1.0))
        max_angle = _env_float("ROBOT_PLACE_VERIFY_MAX_ANGLE_DEG", 35.0)
        geometry_reachable = distance <= max_distance and angle_abs <= max_angle
        interactable = self._object_interactable_from_current_pose(held_object_id)
        interactable_reachable = bool(interactable.get("interactable_from_current_pose", False))
        reachable = bool(geometry_reachable or interactable_reachable)
        target_tolerance_m = max(
            0.0,
            _env_float(
                "ROBOT_PLACE_VERIFY_TARGET_MAX_ERROR_M",
                _env_float("ROBOT_PLACE_EXACT_RESOLVE_MAX_ERROR_M", 0.10),
            ),
        )
        placement_target_error_m: Optional[float] = None
        placement_target_verified = not bool(exact_target_required)
        if exact_target_required and isinstance(expected_surface_world_target, dict):
            object_position = placed_object.get("position") if isinstance(placed_object.get("position"), dict) else {}
            try:
                placement_target_error_m = math.hypot(
                    float(object_position.get("x")) - float(expected_surface_world_target.get("x")),
                    float(object_position.get("z")) - float(expected_surface_world_target.get("z")),
                )
            except (TypeError, ValueError):
                placement_target_error_m = None
            placement_target_verified = bool(
                placement_target_error_m is not None
                and math.isfinite(placement_target_error_m)
                and placement_target_error_m <= target_tolerance_m
            )

        validation: JsonDict = {
            "placement_verified": bool(on_receptacle and reachable and placement_target_verified),
            "object_on_target_receptacle": bool(on_receptacle),
            "object_reachable_from_agent": bool(reachable),
            "object_geometry_reachable_from_agent": bool(geometry_reachable),
            "object_interactable_from_agent": bool(interactable_reachable),
            "interactable_pose_validation": interactable,
            "distance_bucket": self._distance_bucket(distance),
            "angle_bucket": self._angle_bucket(angle_abs),
            "max_distance_m": round(max_distance, 3),
            "max_angle_deg": round(max_angle, 1),
            "placement_target_required": bool(exact_target_required),
            "placement_target_verified": bool(placement_target_verified),
            "placement_target_error_m": (
                round(float(placement_target_error_m), 4)
                if placement_target_error_m is not None
                else None
            ),
            "placement_target_tolerance_m": round(float(target_tolerance_m), 4),
            "placed_object": self._summarize_service_object(placed_object, task_semantic_class="placed_object"),
            "receptacle_after": self._summarize_service_object(receptacle_after, task_semantic_class="place_receptacle"),
        }
        if not on_receptacle:
            validation.update(
                {
                    "result_type": "error_place_receptacle_mismatch",
                    "message": "Place action succeeded, but the object is not on the grounded target receptacle.",
                }
            )
        elif not reachable:
            validation.update(
                {
                    "result_type": "error_place_position_unreachable",
                    "message": "Place action succeeded, but the final object position is outside the reachable front placement zone.",
                }
            )
        elif not placement_target_verified:
            validation.update(
                {
                    "result_type": "error_place_target_deviation",
                    "message": "Place action succeeded, but the final object position is outside the selected RGB-D target tolerance.",
                }
            )
        else:
            validation.update(
                {
                    "result_type": "place_verified_exact_target" if exact_target_required else "place_verified_reachable",
                    "message": (
                        "Placed object is on the target receptacle within the selected RGB-D target tolerance."
                        if exact_target_required
                        else "Placed object is on the target receptacle and still reachable/interactable from the current robot pose."
                    ),
                }
            )
        return validation

    def _object_interactable_from_current_pose(self, object_id: Any) -> JsonDict:
        """ALFRED-style postcondition: verify the current pose can interact with the placed object."""
        if not object_id:
            return {
                "available": False,
                "interactable_from_current_pose": False,
                "reason": "missing_object_id",
                "interactable_pose_count": 0,
            }

        try:
            event = self.controller.step(action="GetInteractablePoses", objectId=str(object_id))
        except Exception as exc:
            return {
                "available": False,
                "interactable_from_current_pose": False,
                "reason": "get_interactable_poses_exception",
                "error_message": str(exc),
                "interactable_pose_count": 0,
            }
        self.last_event = event

        metadata = event.metadata if hasattr(event, "metadata") and isinstance(event.metadata, dict) else {}
        if not bool(metadata.get("lastActionSuccess", False)):
            return {
                "available": False,
                "interactable_from_current_pose": False,
                "reason": "get_interactable_poses_failed",
                "error_message": metadata.get("errorMessage", ""),
                "interactable_pose_count": 0,
            }

        poses = metadata.get("actionReturn") or []
        poses = poses if isinstance(poses, list) else []
        agent = metadata.get("agent") if isinstance(metadata.get("agent"), dict) else {}
        matched_pose = self._match_current_agent_interactable_pose(agent, poses)
        return {
            "available": True,
            "interactable_from_current_pose": matched_pose is not None,
            "reason": "current_pose_interactable" if matched_pose is not None else "current_pose_not_in_interactable_poses",
            "interactable_pose_count": len(poses),
            "matched_pose": matched_pose,
        }

    def _object_interactable_navigation_guidance(self, object_id: Any) -> JsonDict:
        """Return an online-safe action hint toward an AI2-THOR interactable pose."""
        if not object_id:
            return {
                "available": False,
                "current_pose_interactable": False,
                "reason": "missing_object_id",
                "interactable_pose_count": 0,
                "recommended_action": None,
            }

        try:
            event = self.controller.step(action="GetInteractablePoses", objectId=str(object_id))
        except Exception as exc:
            return {
                "available": False,
                "current_pose_interactable": False,
                "reason": "get_interactable_poses_exception",
                "error_message": str(exc),
                "interactable_pose_count": 0,
                "recommended_action": None,
            }
        self.last_event = event

        metadata = event.metadata if hasattr(event, "metadata") and isinstance(event.metadata, dict) else {}
        if not bool(metadata.get("lastActionSuccess", False)):
            return {
                "available": False,
                "current_pose_interactable": False,
                "reason": "get_interactable_poses_failed",
                "error_message": metadata.get("errorMessage", ""),
                "interactable_pose_count": 0,
                "recommended_action": None,
            }

        poses = metadata.get("actionReturn") or []
        poses = poses if isinstance(poses, list) else []
        agent = metadata.get("agent") if isinstance(metadata.get("agent"), dict) else {}
        matched_pose = self._match_current_agent_interactable_pose(agent, poses)
        if matched_pose is not None:
            return {
                "available": True,
                "current_pose_interactable": True,
                "reason": "current_pose_interactable",
                "interactable_pose_count": len(poses),
                "recommended_action": None,
                "distance_bucket": "near",
                "angle_bucket": "front-center",
            }

        nearest = self._nearest_interactable_pose(agent, poses)
        if nearest is None:
            return {
                "available": bool(poses),
                "current_pose_interactable": False,
                "reason": "no_usable_interactable_pose",
                "interactable_pose_count": len(poses),
                "recommended_action": None,
            }

        return self._interactable_pose_action_hint(agent, nearest, len(poses))

    def _nearest_interactable_pose(self, agent: JsonDict, poses: List[JsonDict]) -> Optional[JsonDict]:
        position = agent.get("position") if isinstance(agent.get("position"), dict) else {}
        rotation = agent.get("rotation") if isinstance(agent.get("rotation"), dict) else {}
        try:
            agent_x = float(position.get("x"))
            agent_z = float(position.get("z"))
            agent_rot = float(rotation.get("y", 0.0)) % 360.0
        except (TypeError, ValueError):
            return None

        def angle_delta(a: float, b: float) -> float:
            return (a - b + 180.0) % 360.0 - 180.0

        best: Optional[Tuple[float, JsonDict]] = None
        for pose in poses:
            if not isinstance(pose, dict):
                continue
            try:
                pose_x = float(pose.get("x"))
                pose_z = float(pose.get("z"))
                pose_rot = float(pose.get("rotation", agent_rot)) % 360.0
            except (TypeError, ValueError):
                continue
            distance = math.hypot(pose_x - agent_x, pose_z - agent_z)
            rot_penalty = abs(angle_delta(pose_rot, agent_rot)) / 180.0
            score = distance + 0.08 * rot_penalty
            if best is None or score < best[0]:
                best = (score, pose)
        return dict(best[1]) if best is not None else None

    def _interactable_pose_action_hint(self, agent: JsonDict, pose: JsonDict, pose_count: int) -> JsonDict:
        position = agent.get("position") if isinstance(agent.get("position"), dict) else {}
        rotation = agent.get("rotation") if isinstance(agent.get("rotation"), dict) else {}
        try:
            agent_x = float(position.get("x"))
            agent_z = float(position.get("z"))
            agent_rot = float(rotation.get("y", 0.0)) % 360.0
            pose_x = float(pose.get("x"))
            pose_z = float(pose.get("z"))
            pose_rot = float(pose.get("rotation", agent_rot)) % 360.0
        except (TypeError, ValueError):
            return {
                "available": False,
                "current_pose_interactable": False,
                "reason": "invalid_interactable_pose",
                "interactable_pose_count": pose_count,
                "recommended_action": None,
            }

        def signed_delta(a: float, b: float) -> float:
            return (a - b + 180.0) % 360.0 - 180.0

        dx = pose_x - agent_x
        dz = pose_z - agent_z
        distance = math.hypot(dx, dz)
        target_angle = math.degrees(math.atan2(dx, dz)) % 360.0
        move_delta = signed_delta(target_angle, agent_rot)
        rot_delta = signed_delta(pose_rot, agent_rot)
        pos_tol = _env_float("ROBOT_INTERACTABLE_POSE_APPROACH_TOL", 0.08)
        align_tol = _env_float("ROBOT_INTERACTABLE_POSE_ALIGN_TOL_DEG", 10.0)
        forward_cone = _env_float("ROBOT_INTERACTABLE_POSE_FORWARD_CONE_DEG", 25.0)

        if distance > pos_tol:
            if abs(move_delta) > forward_cone:
                action = "RotateLeft" if move_delta < 0 else "RotateRight"
                reason = "turn_toward_nearest_interactable_pose"
            else:
                action = "MoveAhead"
                reason = "approach_nearest_interactable_pose"
            angle_abs = abs(move_delta)
        elif abs(rot_delta) > align_tol:
            action = "RotateLeft" if rot_delta < 0 else "RotateRight"
            reason = "align_to_nearest_interactable_pose"
            angle_abs = abs(rot_delta)
        else:
            action = None
            reason = "nearest_pose_requires_unavailable_camera_or_stance_adjustment"
            angle_abs = 0.0

        return {
            "available": True,
            "current_pose_interactable": False,
            "reason": reason,
            "interactable_pose_count": pose_count,
            "recommended_action": action,
            "distance_bucket": self._distance_bucket(distance),
            "angle_bucket": self._angle_bucket(angle_abs),
        }

    def _match_current_agent_interactable_pose(
        self,
        agent: JsonDict,
        poses: List[JsonDict],
    ) -> Optional[JsonDict]:
        if not isinstance(agent, dict) or not isinstance(poses, list):
            return None

        position = agent.get("position") if isinstance(agent.get("position"), dict) else {}
        rotation = agent.get("rotation") if isinstance(agent.get("rotation"), dict) else {}
        try:
            agent_x = float(position.get("x"))
            agent_y = float(position.get("y", 0.0))
            agent_z = float(position.get("z"))
            agent_rot = float(rotation.get("y", 0.0)) % 360.0
            agent_horizon = float(agent.get("cameraHorizon", 0.0))
        except (TypeError, ValueError):
            return None
        agent_standing = bool(agent.get("isStanding", True))

        pos_tol = _env_float("ROBOT_INTERACTABLE_POSE_POSITION_TOL", 0.06)
        y_tol = _env_float("ROBOT_INTERACTABLE_POSE_Y_TOL", 0.08)
        rot_tol = _env_float("ROBOT_INTERACTABLE_POSE_ROTATION_TOL_DEG", 5.0)
        horizon_tol = _env_float("ROBOT_INTERACTABLE_POSE_HORIZON_TOL_DEG", 5.0)

        def angle_delta(a: float, b: float) -> float:
            delta = (a - b + 180.0) % 360.0 - 180.0
            return abs(delta)

        for pose in poses:
            if not isinstance(pose, dict):
                continue
            try:
                pose_x = float(pose.get("x"))
                pose_y = float(pose.get("y", agent_y))
                pose_z = float(pose.get("z"))
                pose_rot = float(pose.get("rotation", agent_rot)) % 360.0
                pose_horizon = float(pose.get("horizon", agent_horizon))
            except (TypeError, ValueError):
                continue
            pose_standing = bool(pose.get("standing", agent_standing))
            if pose_standing != agent_standing:
                continue
            if math.hypot(agent_x - pose_x, agent_z - pose_z) > pos_tol:
                continue
            if abs(agent_y - pose_y) > y_tol:
                continue
            if angle_delta(agent_rot, pose_rot) > rot_tol:
                continue
            if abs(agent_horizon - pose_horizon) > horizon_tol:
                continue
            return {
                "x": round(pose_x, 3),
                "y": round(pose_y, 3),
                "z": round(pose_z, 3),
                "rotation": round(pose_rot, 1),
                "horizon": round(pose_horizon, 1),
                "standing": pose_standing,
            }
        return None

    def _relative_angle_delta(self, obj: JsonDict) -> float:
        agent = self.last_event.metadata["agent"]
        agent_pos = agent["position"]
        agent_rot_y = float(agent["rotation"]["y"])
        obj_pos = obj.get("position", {})
        dx = float(obj_pos.get("x", 0.0)) - float(agent_pos.get("x", 0.0))
        dz = float(obj_pos.get("z", 0.0)) - float(agent_pos.get("z", 0.0))
        target_angle = math.degrees(math.atan2(dx, dz))
        delta = target_angle - agent_rot_y
        while delta > 180:
            delta -= 360
        while delta < -180:
            delta += 360
        return delta

    def _relative_position_hint(self, obj: JsonDict) -> str:
        delta = self._relative_angle_delta(obj)
        if abs(delta) <= 15:
            return "front-center"
        if -60 <= delta < -15:
            return "front-left"
        if 15 < delta <= 60:
            return "front-right"
        if delta < -60:
            return "left-or-behind"
        return "right-or-behind"

    def _is_floor_level_object(self, obj: JsonDict) -> bool:
        pos = obj.get("position", {})
        y = pos.get("y", None)
        if y is None:
            return False
        return float(y) <= float(os.getenv("ROBOT_FLOOR_Y_THRESHOLD", "0.35"))

    def _robot_state_changed(self, before: JsonDict, after: JsonDict) -> bool:
        before_pos = before.get("position") or {}
        after_pos = after.get("position") or {}
        before_rot = before.get("rotation") or {}
        after_rot = after.get("rotation") or {}
        pos_changed = any(
            abs(float(before_pos.get(k, 0.0)) - float(after_pos.get(k, 0.0))) > 1e-4
            for k in ("x", "y", "z")
        )
        rot_changed = any(
            abs(float(before_rot.get(k, 0.0)) - float(after_rot.get(k, 0.0))) > 1e-4
            for k in ("x", "y", "z")
        )
        horizon_changed = abs(
            float(before.get("cameraHorizon", 0.0) or 0.0)
            - float(after.get("cameraHorizon", 0.0) or 0.0)
        ) > 1e-4
        return pos_changed or rot_changed or horizon_changed
