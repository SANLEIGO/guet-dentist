from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DENTAL_TEMPLATE_DIR = PROJECT_ROOT / "assets" / "dental_templates"


@dataclass(frozen=True)
class DentalTemplateAsset:
    asset_id: str
    title: str
    model_path: Path
    license_path: Path
    source_url: str
    author: str
    license_name: str
    notes: str = ""

    @property
    def exists(self) -> bool:
        return self.model_path.exists()

    @property
    def model_size_mb(self) -> float:
        if not self.model_path.exists():
            return 0.0
        return self.model_path.stat().st_size / (1024 * 1024)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["model_path"] = str(self.model_path)
        payload["license_path"] = str(self.license_path)
        payload["model_size_mb"] = round(self.model_size_mb, 2)
        return payload


def get_default_dental_arch_asset() -> DentalTemplateAsset:
    return DentalTemplateAsset(
        asset_id="dental_arches_sketchfab_cc_by_v1",
        title="Dental arches",
        model_path=DENTAL_TEMPLATE_DIR / "dental_arches.glb",
        license_path=DENTAL_TEMPLATE_DIR / "dental_arches_LICENSE.txt",
        source_url="https://sketchfab.com/3d-models/dental-arches-a17fda74c85344709624e9e39a6634b6",
        author="",
        license_name="CC-BY-4.0",
        notes="Complete upper/lower dental arch visual baseline. Meshes are not separated by FDI tooth.",
    )


@lru_cache(maxsize=2)
def load_model_data_uri(model_path: str) -> str:
    path = Path(model_path)
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:model/gltf-binary;base64,{encoded}"


def read_asset_license(asset: DentalTemplateAsset) -> str:
    if not asset.license_path.exists():
        return ""
    return asset.license_path.read_text(encoding="utf-8").strip()
