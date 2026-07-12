#!/usr/bin/env python3
"""Fast contract checks for model-pinned, task-specific CLIs."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
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
    parser.add_argument("--texturing", required=True, type=Path)
    parser.add_argument("--tokenskin", required=True, type=Path)
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

    texturing_help = run(args.texturing, "--help")
    require(texturing_help.returncode == 0,
            "TRELLIS.2 mesh texturing --help failed", texturing_help.stdout)
    require("TRELLIS.2 shape-conditioned material pipeline" in texturing_help.stdout,
            "wrong TRELLIS.2 mesh texturing task banner", texturing_help.stdout)
    for required_flag in ("--input", "--image", "--shape-encoder", "--texture-flow"):
        require(required_flag in texturing_help.stdout,
                f"TRELLIS.2 mesh texturing is missing {required_flag}",
                texturing_help.stdout)
    for unrelated_flag in ("--naf", "--num-beams", "--mesh-remesh"):
        require(unrelated_flag not in texturing_help.stdout,
                f"TRELLIS.2 mesh texturing exposes {unrelated_flag}",
                texturing_help.stdout)

    wrong_texturing_family = run(
        args.texturing,
        "--model",
        str(args.source_dir / "models" / "pixal3d"),
        "--dino",
        "must-not-be-opened-dino",
        "--input",
        "must-not-be-opened.glb",
        "--image",
        "must-not-be-opened.png",
        "--output",
        "must-not-be-written.glb",
    )
    require(wrong_texturing_family.returncode == 1,
            "TRELLIS.2 mesh texturing accepted the Pixal3D package",
            wrong_texturing_family.stdout)
    require("is not trellis2" in wrong_texturing_family.stdout,
            "TRELLIS.2 mesh texturing family error is unclear",
            wrong_texturing_family.stdout)
    require("ggml_cuda_init" not in wrong_texturing_family.stdout and
            "ggml_vulkan" not in wrong_texturing_family.stdout,
            "TRELLIS.2 mesh texturing initialized a GPU before family rejection",
            wrong_texturing_family.stdout)

    tokenskin_help = run(args.tokenskin, "--help")
    require(tokenskin_help.returncode == 0,
            "TokenSkin --help failed", tokenskin_help.stdout)
    require("TokenSkin/TokenRig mesh rigging" in tokenskin_help.stdout,
            "wrong TokenSkin task banner", tokenskin_help.stdout)
    for required_flag in ("--input", "--num-beams", "--official-eos-compat"):
        require(required_flag in tokenskin_help.stdout,
                f"TokenSkin is missing {required_flag}", tokenskin_help.stdout)
    for image_flag in ("--image", "--dino", "--naf", "--pipeline"):
        require(image_flag not in tokenskin_help.stdout,
                f"TokenSkin exposes image-to-3D flag {image_flag}", tokenskin_help.stdout)

    wrong_tokenskin_family = run(
        args.tokenskin,
        "--model",
        str(args.source_dir / "models" / "trellis2"),
        "--input",
        "must-not-be-opened.glb",
        "--output",
        "must-not-be-written.glb",
    )
    require(wrong_tokenskin_family.returncode == 1,
            "TokenSkin did not reject the TRELLIS.2 package",
            wrong_tokenskin_family.stdout)
    require("requires family=tokenskin task=mesh_rigging" in wrong_tokenskin_family.stdout,
            "TokenSkin family/task error is unclear", wrong_tokenskin_family.stdout)
    require("must-not-be-opened" not in wrong_tokenskin_family.stdout and
            "ggml_cuda_init" not in wrong_tokenskin_family.stdout and
            "ggml_vulkan" not in wrong_tokenskin_family.stdout,
            "TokenSkin parsed the mesh or initialized a GPU before family rejection",
            wrong_tokenskin_family.stdout)

    with tempfile.TemporaryDirectory() as temporary_directory:
        invalid_package = Path(temporary_directory)
        manifest = json.loads(
            (args.source_dir / "models" / "tokenskin" / "model.json").read_text()
        )
        manifest["components"][1]["execution"]["compute_dtype"] = "f16"
        (invalid_package / "model.json").write_text(json.dumps(manifest))
        wrong_tokenskin_policy = run(
            args.tokenskin,
            "--model",
            str(invalid_package),
            "--input",
            "must-not-be-opened.glb",
            "--output",
            "must-not-be-written.glb",
        )
        require(wrong_tokenskin_policy.returncode == 1,
                "TokenSkin accepted an incompatible execution policy",
                wrong_tokenskin_policy.stdout)
        require("declare compute_dtype=bf16 attention=flash flash_kv_dtype=bf16" in
                wrong_tokenskin_policy.stdout,
                "TokenSkin execution-policy error is unclear",
                wrong_tokenskin_policy.stdout)
        require("must-not-be-opened" not in wrong_tokenskin_policy.stdout and
                "ggml_cuda_init" not in wrong_tokenskin_policy.stdout and
                "ggml_vulkan" not in wrong_tokenskin_policy.stdout,
                "TokenSkin initialized input/backend before policy rejection",
                wrong_tokenskin_policy.stdout)

    print("model CLI split tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
