"""Near-duplicate grouping utilities shared by the dataset-preparation script and
the (optional) grouped train/validation split.

Policy, in one place so prepare_leakage_controlled_dataset.py and any future
caller agree on it:

- Build an undirected graph over image pairs reported in a near-duplicate
  candidate CSV (audit_near_duplicates.py's output) at or below a chosen
  Hamming-distance threshold, using each image's resolved filesystem path as
  the node identity rather than the CSV's own "split" label. The CSV's split
  labels reflect one specific seed/validation_split realization of
  split_files_by_class; a near-duplicate relationship between two physical
  files does not depend on which virtual partition either one happened to
  land in during that one audit run.
- Connected components are computed with union-find. A component that
  includes any image physically located under a Testing/ folder is a
  "test-touching" component: every one of its Training/-folder members is a
  leakage-control exclusion candidate (Testing/ itself is never modified).
- A component with no Testing/ member is a "pure training" component: all of
  its members must stay on the same side of a future train/validation split.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path


class UnionFind:
    """Minimal union-find over arbitrary hashable keys."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, key: str) -> str:
        self._parent.setdefault(key, key)
        root = key
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[key] != root:
            self._parent[key], key = root, self._parent[key]
        return root

    def union(self, a: str, b: str) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self._parent[root_a] = root_b

    def nodes(self) -> list[str]:
        return list(self._parent)


@dataclass
class DuplicateGroups:
    """Connected components over a near-duplicate candidate CSV."""

    max_distance: int
    components: list[set[str]] = field(default_factory=list)
    testing_root: str = ""

    def touches_testing(self, component: set[str]) -> bool:
        return any(path.startswith(self.testing_root) for path in component)

    def training_members(self, component: set[str]) -> set[str]:
        return {path for path in component if not path.startswith(self.testing_root)}

    def excluded_training_paths(self) -> dict[str, int]:
        """Training-folder paths that must be excluded, mapped to their group index."""
        excluded: dict[str, int] = {}
        for index, component in enumerate(self.components):
            if self.touches_testing(component):
                for path in self.training_members(component):
                    excluded[path] = index
        return excluded

    def pure_training_groups(self) -> dict[str, list[str]]:
        """group_id -> sorted member paths, for components with 2+ members and no Testing/ image."""
        groups: dict[str, list[str]] = {}
        for index, component in enumerate(self.components):
            if not self.touches_testing(component) and len(component) >= 2:
                groups[f"group_{index}"] = sorted(component)
        return groups


def build_duplicate_groups(
    near_duplicates_csv: Path,
    testing_root: Path,
    max_distance: int = 0,
) -> DuplicateGroups:
    """Read a near_duplicate_candidates.csv and compute connected components.

    Only pairs with hamming_distance <= max_distance are used as edges. Node
    identity is each image's resolved absolute path, independent of the CSV's
    own split labels (see module docstring).
    """
    union_find = UnionFind()
    with near_duplicates_csv.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if int(row["hamming_distance"]) > max_distance:
                continue
            path_a = str(Path(row["image_1"]).resolve())
            path_b = str(Path(row["image_2"]).resolve())
            union_find.union(path_a, path_b)

    members_by_root: dict[str, set[str]] = {}
    for node in union_find.nodes():
        root = union_find.find(node)
        members_by_root.setdefault(root, set()).add(node)

    return DuplicateGroups(
        max_distance=max_distance,
        components=list(members_by_root.values()),
        testing_root=str(testing_root.resolve()),
    )
