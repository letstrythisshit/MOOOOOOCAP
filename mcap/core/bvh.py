from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class BvhJoint:
    name: str
    parent: str | None
    offset: np.ndarray


@dataclass
class BvhFrame:
    root_position: np.ndarray
    rotations_deg: Dict[str, np.ndarray]


class BvhWriter:
    def __init__(self, joints: List[BvhJoint]) -> None:
        self.joints = joints
        self._children = {joint.name: [] for joint in joints}
        for joint in joints:
            if joint.parent:
                self._children[joint.parent].append(joint.name)

    def _write_joint(self, lines: List[str], joint_name: str, indent: int) -> None:
        joint = next(j for j in self.joints if j.name == joint_name)
        indent_str = "\t" * indent
        keyword = "ROOT" if joint.parent is None else "JOINT"
        lines.append(f"{indent_str}{keyword} {joint.name}")
        lines.append(f"{indent_str}{{")
        offset = joint.offset
        lines.append(
            f"{indent_str}\tOFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}"
        )
        if joint.parent is None:
            lines.append(
                f"{indent_str}\tCHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation"
            )
        else:
            lines.append(f"{indent_str}\tCHANNELS 3 Zrotation Xrotation Yrotation")

        for child in self._children[joint_name]:
            self._write_joint(lines, child, indent + 1)

        if not self._children[joint_name]:
            lines.append(f"{indent_str}\tEnd Site")
            lines.append(f"{indent_str}\t{{")
            lines.append(f"{indent_str}\t\tOFFSET 0.0 0.0 0.0")
            lines.append(f"{indent_str}\t}}")
        lines.append(f"{indent_str}}}")

    def write(self, frames: List[BvhFrame], fps: int) -> str:
        lines: List[str] = ["HIERARCHY"]
        root = next(j.name for j in self.joints if j.parent is None)
        self._write_joint(lines, root, 0)
        lines.append("MOTION")
        lines.append(f"Frames: {len(frames)}")
        lines.append(f"Frame Time: {1.0 / fps:.6f}")
        for frame in frames:
            channels = []
            root_position = frame.root_position
            channels.extend(
                [
                    f"{root_position[0]:.6f}",
                    f"{root_position[1]:.6f}",
                    f"{root_position[2]:.6f}",
                ]
            )
            root_rot = frame.rotations_deg.get(root, np.zeros(3))
            channels.extend(
                [f"{root_rot[2]:.6f}", f"{root_rot[0]:.6f}", f"{root_rot[1]:.6f}"]
            )
            for joint in self.joints:
                if joint.parent is None:
                    continue
                rot = frame.rotations_deg.get(joint.name, np.zeros(3))
                channels.extend(
                    [f"{rot[2]:.6f}", f"{rot[0]:.6f}", f"{rot[1]:.6f}"]
                )
            lines.append(" ".join(channels))
        return "\n".join(lines) + "\n"
