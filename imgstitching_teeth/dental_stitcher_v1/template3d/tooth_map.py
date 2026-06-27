from __future__ import annotations

from dental_stitcher_v1.template3d.schema import ArchLabel, TemplateTooth


_FDI_QUADRANTS_BY_ARCH = {
    ArchLabel.UPPER: (1, 2),
    ArchLabel.LOWER: (4, 3),
}


def build_adult_template_teeth(arch_label: ArchLabel) -> list[TemplateTooth]:
    if arch_label not in _FDI_QUADRANTS_BY_ARCH:
        return []

    teeth: list[TemplateTooth] = []
    for quadrant in _FDI_QUADRANTS_BY_ARCH[arch_label]:
        for position in range(1, 9):
            tooth_id = quadrant * 10 + position
            teeth.append(
                TemplateTooth(
                    tooth_id=tooth_id,
                    arch_label=arch_label,
                    quadrant=quadrant,
                    position=position,
                    display_name=f"FDI {tooth_id}",
                    mesh_node_name=f"tooth_{tooth_id}",
                )
            )
    return teeth

