"""Visual language for PruneMaxxing showcase renders."""

from __future__ import annotations

# Soil / canopy / amber robotics — avoid generic purple AI look
COLORS = {
    "bg": "#0f1410",
    "panel": "#1a2218",
    "ink": "#e8efe4",
    "muted": "#9aab94",
    "soil": "#6b4a2e",
    "canopy": "#3d6b3a",
    "amber": "#e0972f",
    "cyan": "#3db8c5",
    "weed": "#c44536",
    "planned": "#f0d78c",
    "actual": "#3db8c5",
    "drone": "#f3f6f0",
    "completion": "YlGn",
}

# Per-drone path hues: (planned dashed, actual solid) — same family, distinct drones
DRONE_PATH_COLORS: list[tuple[str, str]] = [
    ("#c4b5fd", "#a78bfa"),  # violet (avoid MS cyan clash)
    ("#ffd27a", "#ff9f1a"),  # bright amber
    ("#b8f26d", "#7CFC00"),  # lime
    ("#ff9ecd", "#ff2d95"),  # magenta
]

FONT_FAMILY = "DejaVu Sans"
TITLE_SIZE = 16
LABEL_SIZE = 10


def drone_path_colors(drone_id: int) -> tuple[str, str]:
    return DRONE_PATH_COLORS[drone_id % len(DRONE_PATH_COLORS)]
