#!/usr/bin/env python3
"""Fast contract checks for the model-pinned image-to-3D CLIs."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(executable: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(executable), *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def require(condition: bool, message: str, output: str = "") -> None:
    if condition:
        return
    if output:
        message = f"{message}\n--- command output ---\n{output}"
    raise AssertionError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trellis2", required=True, type=Path)
    parser.add_argument("--pixal3d", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    args = parser.parse_args()

    trellis_help = run(args.trellis2, "--help")
    require(trellis_help.returncode == 0, "TRELLIS.2 --help failed", trellis_help.stdout)
    require("TRELLIS.2 image-to-3D" in trellis_help.stdout, "wrong TRELLIS.2 banner", trellis_help.stdout)
    require("Profile: fixed 512." in trellis_help.stdout, "TRELLIS.2 profile is not fixed", trellis_help.stdout)
    for pixal_flag in ("--naf", "--pipeline", "--fov", "--camera-distance", "--mesh-scale"):
        require(pixal_flag not in trellis_help.stdout, f"TRELLIS.2 exposes {pixal_flag}", trellis_help.stdout)

    pixal_help = run(args.pixal3d, "--help")
    require(pixal_help.returncode == 0, "Pixal3D --help failed", pixal_help.stdout)
    require("Pixal3D image-to-3D" in pixal_help.stdout, "wrong Pixal3D banner", pixal_help.stdout)
    for pixal_flag in ("--naf", "--pipeline", "--fov", "--camera-distance", "--mesh-scale"):
        require(pixal_flag in pixal_help.stdout, f"Pixal3D is missing {pixal_flag}", pixal_help.stdout)

    rejected_flag = run(args.trellis2, "--naf", "unused.safetensors")
    require(rejected_flag.returncode == 2, "TRELLIS.2 accepted --naf", rejected_flag.stdout)

    rejected_profile = run(args.pixal3d, "--pipeline", "512")
    require(rejected_profile.returncode == 2, "Pixal3D accepted non-cascade pipeline", rejected_profile.stdout)
    require("must be 1024_cascade or 1536_cascade" in rejected_profile.stdout,
            "Pixal3D profile error is unclear", rejected_profile.stdout)

    common = ("--dino", "unused-dino", "--image", "unused.png")
    wrong_trellis_family = run(
        args.trellis2,
        "--model",
        str(args.source_dir / "models" / "pixal3d"),
        *common,
    )
    require(wrong_trellis_family.returncode == 1,
            "TRELLIS.2 did not reject the Pixal3D package", wrong_trellis_family.stdout)
    require("requires family 'trellis2'" in wrong_trellis_family.stdout,
            "TRELLIS.2 family error is unclear", wrong_trellis_family.stdout)

    wrong_pixal_family = run(
        args.pixal3d,
        "--model",
        str(args.source_dir / "models" / "trellis2"),
        *common,
    )
    require(wrong_pixal_family.returncode == 1,
            "Pixal3D did not reject the TRELLIS.2 package", wrong_pixal_family.stdout)
    require("requires family 'pixal3d'" in wrong_pixal_family.stdout,
            "Pixal3D family error is unclear", wrong_pixal_family.stdout)

    print("model CLI split tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
