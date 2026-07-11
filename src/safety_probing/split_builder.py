from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .config import DatasetConfig
from .io_utils import load_json, write_json


SPLIT_KEYS = {
    "train": "train_pair_ids",
    "val": "val_pair_ids",
    "test": "test_pair_ids",
}


class UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _rounded(value: Any) -> float:
    return round(float(value), 9)


def _payload_state(
    scene: dict[str, Any], intervention_type: str, intervention_value: float
) -> tuple[float, tuple[float, float, float], float]:
    mass_kg = float(scene["load_base_mass_kg"])
    com_xyz = [float(value) for value in scene["load_base_com_offset_local_xyz_m"]]
    load_height = float(scene["load_spawn_lift_m"])

    if intervention_type == "load_mass":
        mass_kg = intervention_value
    elif intervention_type == "com_shift":
        com_xyz[1] = intervention_value
    elif intervention_type == "com_forward_shift":
        com_xyz[0] = intervention_value
    elif intervention_type == "load_height":
        load_height = intervention_value

    return (
        _rounded(mass_kg),
        tuple(_rounded(value) for value in com_xyz),
        _rounded(load_height),
    )


def _condition_key(
    pair: dict[str, Any], scene: dict[str, Any], intervention_value: float
) -> tuple[Any, ...]:
    return (
        str(pair["route_id"]),
        int(pair["seed"]),
        *_payload_state(scene, str(pair["intervention_type"]), intervention_value),
        tuple(_rounded(value) for value in scene["load_size_m"]),
        tuple(_rounded(value) for value in scene["load_base_offset_m"]),
    )


def _mirror_pair_key(
    pair: dict[str, Any], route: dict[str, Any], scene: dict[str, Any]
) -> tuple[Any, ...]:
    return (
        str(pair["intervention_type"]),
        _rounded(pair["base_value"]),
        _rounded(pair["modified_value"]),
        int(pair["seed"]),
        str(route.get("mirror_group_id")),
        str(route.get("speed_label")),
        str(route.get("steering_label")),
        _rounded(scene["load_base_mass_kg"]),
        tuple(_rounded(value) for value in scene["load_base_com_offset_local_xyz_m"]),
        _rounded(scene["load_spawn_lift_m"]),
    )


def _hash_key(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


class SplitSource:
    def __init__(self, source_config: dict[str, Any]) -> None:
        self.source_config = source_config
        self.pairs = {
            str(pair["pair_id"]): pair for pair in source_config["pair_requests"]
        }
        self.routes = {str(route["route_id"]): route for route in source_config["routes"]}
        self.scenes = {
            str(scene["scenario_id"]): scene for scene in source_config["scenes"]
        }
        if len(self.pairs) != len(source_config["pair_requests"]):
            raise ValueError("Duplicate pair_id values in source config")

        self.endpoint_conditions: dict[str, tuple[tuple[Any, ...], tuple[Any, ...]]] = {}
        self.condition_to_pairs: dict[tuple[Any, ...], list[str]] = defaultdict(list)
        self.mirror_to_pairs: dict[tuple[Any, ...], list[str]] = defaultdict(list)
        for pair_id, pair in self.pairs.items():
            scene = self.scenes[str(pair["scenario_id"])]
            route = self.routes[str(pair["route_id"])]
            endpoints = (
                _condition_key(pair, scene, float(pair["base_value"])),
                _condition_key(pair, scene, float(pair["modified_value"])),
            )
            self.endpoint_conditions[pair_id] = endpoints
            for endpoint in set(endpoints):
                self.condition_to_pairs[endpoint].append(pair_id)
            self.mirror_to_pairs[_mirror_pair_key(pair, route, scene)].append(pair_id)

    def direction(self, pair_id: str) -> str:
        route_id = str(self.pairs[pair_id]["route_id"])
        return str(self.routes[route_id].get("turn_direction"))

    def components(self, pair_ids: set[str], *, include_mirrors: bool) -> list[list[str]]:
        union_find = UnionFind(pair_ids)
        for members in self.condition_to_pairs.values():
            selected = sorted(pair_id for pair_id in members if pair_id in pair_ids)
            for pair_id in selected[1:]:
                union_find.union(selected[0], pair_id)
        if include_mirrors:
            for members in self.mirror_to_pairs.values():
                selected = sorted(pair_id for pair_id in members if pair_id in pair_ids)
                for pair_id in selected[1:]:
                    union_find.union(selected[0], pair_id)

        groups: dict[str, list[str]] = defaultdict(list)
        for pair_id in sorted(pair_ids):
            groups[union_find.find(pair_id)].append(pair_id)
        return list(groups.values())


def _assign_components(
    components: list[list[str]],
    ratios: dict[str, float],
    *,
    seed: int,
    features_by_pair: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, list[str]]:
    total = sum(len(component) for component in components)
    ratio_sum = sum(ratios.values())
    normalized = {name: ratio / ratio_sum for name, ratio in ratios.items()}
    targets = {name: total * ratio for name, ratio in normalized.items()}
    assigned_components: dict[str, list[list[str]]] = {name: [] for name in ratios}
    counts = {name: 0 for name in ratios}
    raw_component_targets = {
        name: len(components) * ratio for name, ratio in normalized.items()
    }
    component_quotas = {
        name: int(math.floor(target)) for name, target in raw_component_targets.items()
    }
    remaining_components = len(components) - sum(component_quotas.values())
    quota_order = sorted(
        ratios,
        key=lambda name: (-(raw_component_targets[name] - component_quotas[name]), name),
    )
    for name in quota_order[:remaining_components]:
        component_quotas[name] += 1
    assigned_component_counts = {name: 0 for name in ratios}
    feature_totals: dict[str, int] = defaultdict(int)
    component_features: dict[str, dict[str, int]] = {}
    if features_by_pair:
        for component in components:
            counts_for_component: dict[str, int] = defaultdict(int)
            for pair_id in component:
                for feature in features_by_pair[pair_id]:
                    counts_for_component[feature] += 1
                    feature_totals[feature] += 1
            component_features[component[0]] = dict(counts_for_component)
    assigned_features: dict[str, dict[str, int]] = {
        name: defaultdict(int) for name in ratios
    }

    ordered = sorted(
        components,
        key=lambda component: (
            -len(component),
            hashlib.sha256(f"{seed}:{component[0]}".encode("utf-8")).hexdigest(),
        ),
    )
    for component in ordered:
        size = len(component)
        features = component_features.get(component[0], {})

        def assignment_score(candidate: str) -> float:
            size_score = sum(
                (
                    (
                        counts[name]
                        + (size if name == candidate else 0)
                        - targets[name]
                    )
                    / max(targets[name], 1.0)
                )
                ** 2
                for name in ratios
            )
            feature_score = 0.0
            for feature, feature_total in feature_totals.items():
                for name, ratio in normalized.items():
                    target = feature_total * ratio
                    projected = assigned_features[name][feature]
                    if name == candidate:
                        projected += features.get(feature, 0)
                    feature_score += ((projected - target) / max(target, 1.0)) ** 2
            return size_score + 0.35 * feature_score

        candidates = [
            name
            for name in ratios
            if assigned_component_counts[name] < component_quotas[name]
        ]
        best_split = min(candidates, key=assignment_score)
        assigned_components[best_split].append(component)
        counts[best_split] += size
        assigned_component_counts[best_split] += 1
        for feature, count in features.items():
            assigned_features[best_split][feature] += count

    if feature_totals:
        def overall_score(candidate: dict[str, list[list[str]]]) -> float:
            score = 0.0
            for name, components_for_split in candidate.items():
                pair_count = sum(len(component) for component in components_for_split)
                score += 2.0 * ((pair_count - targets[name]) / max(targets[name], 1.0)) ** 2
                observed: Counter[str] = Counter()
                component_presence: Counter[str] = Counter()
                for component in components_for_split:
                    features = component_features.get(component[0], {})
                    observed.update(features)
                    component_presence.update(features.keys())
                for feature, feature_total in feature_totals.items():
                    target = feature_total * normalized[name]
                    score += 0.35 * ((observed[feature] - target) / max(target, 1.0)) ** 2
                    if observed[feature] == 0:
                        score += 10.0
                if component_quotas[name] == 4:
                    desired_speed_components = {
                        "speed:slow": 1,
                        "speed:medium": 2,
                        "speed:fast": 1,
                    }
                    for feature, minimum in desired_speed_components.items():
                        score += 10.0 * max(0, minimum - component_presence[feature])
            return score

        current_score = overall_score(assigned_components)
        split_names = list(ratios)
        for _ in range(100):
            best_swap: tuple[str, int, str, int] | None = None
            best_score = current_score
            for left_index, left_name in enumerate(split_names):
                for right_name in split_names[left_index + 1 :]:
                    for left_component_index in range(len(assigned_components[left_name])):
                        for right_component_index in range(len(assigned_components[right_name])):
                            assigned_components[left_name][left_component_index], assigned_components[
                                right_name
                            ][right_component_index] = (
                                assigned_components[right_name][right_component_index],
                                assigned_components[left_name][left_component_index],
                            )
                            candidate_score = overall_score(assigned_components)
                            assigned_components[left_name][left_component_index], assigned_components[
                                right_name
                            ][right_component_index] = (
                                assigned_components[right_name][right_component_index],
                                assigned_components[left_name][left_component_index],
                            )
                            if candidate_score + 1.0e-12 < best_score:
                                best_score = candidate_score
                                best_swap = (
                                    left_name,
                                    left_component_index,
                                    right_name,
                                    right_component_index,
                                )
            if best_swap is None:
                break
            left_name, left_index, right_name, right_index = best_swap
            assigned_components[left_name][left_index], assigned_components[right_name][right_index] = (
                assigned_components[right_name][right_index],
                assigned_components[left_name][left_index],
            )
            current_score = best_score

    return {
        name: sorted(pair_id for component in components_for_split for pair_id in component)
        for name, components_for_split in assigned_components.items()
    }


def _protocol_payload(description: str, assigned: dict[str, list[str]]) -> dict[str, Any]:
    return {
        "description": description,
        "train_pair_ids": assigned.get("train", []),
        "val_pair_ids": assigned.get("val", []),
        "test_pair_ids": assigned.get("test", []),
    }


def audit_protocol(source: SplitSource, protocol: dict[str, Any]) -> dict[str, Any]:
    pair_to_splits: dict[str, list[str]] = defaultdict(list)
    unknown_pair_ids: list[str] = []
    for split_name, key in SPLIT_KEYS.items():
        for pair_id in protocol.get(key, []):
            if pair_id not in source.pairs:
                unknown_pair_ids.append(pair_id)
            pair_to_splits[pair_id].append(split_name)

    duplicate_pair_assignments = {
        pair_id: splits for pair_id, splits in pair_to_splits.items() if len(splits) > 1
    }
    condition_leaks: list[dict[str, Any]] = []
    for condition, pair_ids in source.condition_to_pairs.items():
        split_names = sorted(
            {
                split_name
                for pair_id in pair_ids
                for split_name in pair_to_splits.get(pair_id, [])
            }
        )
        if len(split_names) > 1:
            condition_leaks.append(
                {
                    "condition_id": _hash_key(condition),
                    "splits": split_names,
                    "pair_ids": sorted(pair_ids),
                }
            )
    condition_leaks.sort(key=lambda row: row["condition_id"])

    mirror_mismatches = 0
    for pair_ids in source.mirror_to_pairs.values():
        split_names = {
            pair_to_splits[pair_id][0]
            for pair_id in pair_ids
            if len(pair_to_splits.get(pair_id, [])) == 1
        }
        if len(split_names) > 1:
            mirror_mismatches += 1

    assigned_pair_ids = set(pair_to_splits)
    return {
        "assigned_pair_count": len(assigned_pair_ids),
        "missing_pair_count": len(set(source.pairs) - assigned_pair_ids),
        "unknown_pair_count": len(set(unknown_pair_ids)),
        "duplicate_pair_assignment_count": len(duplicate_pair_assignments),
        "cross_split_physical_condition_count": len(condition_leaks),
        "mirror_group_split_mismatch_count": mirror_mismatches,
        "cross_split_condition_examples": condition_leaks[:20],
    }


def build_split_manifest(
    source_config: dict[str, Any], reference_manifest: dict[str, Any], *, seed: int = 20260711
) -> dict[str, Any]:
    source = SplitSource(source_config)
    all_pair_ids = set(source.pairs)
    balance_features = {
        pair_id: (
            f"speed:{source.routes[str(pair['route_id'])].get('speed_label')}",
            f"steering:{source.routes[str(pair['route_id'])].get('steering_label')}",
        )
        for pair_id, pair in source.pairs.items()
    }

    main_components = source.components(all_pair_ids, include_mirrors=True)
    main_assigned = _assign_components(
        main_components,
        {"train": 0.70, "val": 0.15, "test": 0.15},
        seed=seed,
        features_by_pair=balance_features,
    )
    main = _protocol_payload(
        "Leakage-safe 70/15/15 split. Pairs sharing a realized physical endpoint are connected, "
        "and left/right mirror counterparts stay in the same split.",
        main_assigned,
    )

    def direction_holdout(train_direction: str, test_direction: str, offset: int) -> dict[str, Any]:
        train_val_ids = {
            pair_id for pair_id in all_pair_ids if source.direction(pair_id) == train_direction
        }
        test_ids = sorted(
            pair_id for pair_id in all_pair_ids if source.direction(pair_id) == test_direction
        )
        components = source.components(train_val_ids, include_mirrors=False)
        assigned = _assign_components(
            components,
            {"train": 0.85, "val": 0.15},
            seed=seed + offset,
            features_by_pair=balance_features,
        )
        assigned["test"] = test_ids
        return _protocol_payload(
            f"Train/validation on {train_direction} turns with endpoint-connected groups; "
            f"evaluate all {test_direction} turns.",
            assigned,
        )

    left_to_right = direction_holdout("left", "right", 1)
    right_to_left = direction_holdout("right", "left", 2)
    protocols = {
        "main": main,
        "left_to_right": left_to_right,
        "right_to_left": right_to_left,
    }

    reference_audit = {
        name: audit_protocol(source, reference_manifest[name])
        for name in protocols
        if name in reference_manifest
    }
    generated_audit = {
        name: audit_protocol(source, protocol) for name, protocol in protocols.items()
    }
    for name, audit in generated_audit.items():
        if audit["cross_split_physical_condition_count"] != 0:
            raise RuntimeError(f"Generated {name} split still leaks physical endpoint conditions")
        if audit["duplicate_pair_assignment_count"] != 0 or audit["missing_pair_count"] != 0:
            raise RuntimeError(f"Generated {name} split does not assign every pair exactly once")

    return {
        "schema_version": 1,
        "source_pair_count": len(source.pairs),
        "split_seed": seed,
        "split_unit": "connected components of shared realized physical endpoint conditions",
        "main_component_count": len(main_components),
        **protocols,
        "audit": {
            "reference_manifest": reference_audit,
            "generated_manifest": generated_audit,
        },
    }


def build_and_write_splits(config: DatasetConfig) -> dict[str, Any]:
    payload = build_split_manifest(
        load_json(config.source_config), load_json(config.reference_split_manifest)
    )
    write_json(config.generated_split_manifest, payload)
    return payload
