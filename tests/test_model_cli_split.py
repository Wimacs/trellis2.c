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
        timeout=30,
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
    parser.add_argument("--segmentation", required=True, type=Path)
    parser.add_argument("--tokenskin", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    args = parser.parse_args()

    trellis_help = run(args.trellis2, "--help")
    require(trellis_help.returncode == 0, "TRELLIS.2 --help failed", trellis_help.stdout)
    require("TRELLIS.2 image-to-3D" in trellis_help.stdout, "wrong TRELLIS.2 banner", trellis_help.stdout)
    require("Profile: 512 by default; --pipeline also accepts 1024." in trellis_help.stdout,
            "TRELLIS.2 pipeline profiles are unclear", trellis_help.stdout)
    require("--pipeline NAME         512 (default) or 1024" in trellis_help.stdout,
            "TRELLIS.2 is missing its pipeline option", trellis_help.stdout)
    require("--prepared-image-output" in trellis_help.stdout,
            "TRELLIS.2 is missing prepared-image persistence", trellis_help.stdout)
    require("--shape-only" in trellis_help.stdout,
            "TRELLIS.2 is missing shape-only generation", trellis_help.stdout)
    require("--shape-latent-output" in trellis_help.stdout,
            "TRELLIS.2 is missing shape latent persistence", trellis_help.stdout)
    for pixal_flag in ("--naf", "--fov", "--camera-distance", "--mesh-scale"):
        require(pixal_flag not in trellis_help.stdout, f"TRELLIS.2 exposes {pixal_flag}", trellis_help.stdout)

    pixal_help = run(args.pixal3d, "--help")
    require(pixal_help.returncode == 0, "Pixal3D --help failed", pixal_help.stdout)
    require("Pixal3D image-to-3D" in pixal_help.stdout, "wrong Pixal3D banner", pixal_help.stdout)
    for pixal_flag in ("--naf", "--pipeline", "--fov", "--camera-distance", "--mesh-scale"):
        require(pixal_flag in pixal_help.stdout, f"Pixal3D is missing {pixal_flag}", pixal_help.stdout)
    require("--prepared-image-output" in pixal_help.stdout,
            "Pixal3D is missing prepared-image persistence", pixal_help.stdout)
    require("--shape-only" in pixal_help.stdout,
            "Pixal3D is missing shape-only generation", pixal_help.stdout)
    require("--shape-latent-output" not in pixal_help.stdout,
            "Pixal3D advertises unsupported shape latent persistence", pixal_help.stdout)

    rejected_pixal_latent = run(args.pixal3d, "--shape-latent-output", "unused.tslat")
    require(rejected_pixal_latent.returncode == 2,
            "Pixal3D accepted unsupported shape latent persistence",
            rejected_pixal_latent.stdout)

    rejected_flag = run(args.trellis2, "--naf", "unused.safetensors")
    require(rejected_flag.returncode == 2, "TRELLIS.2 accepted --naf", rejected_flag.stdout)

    rejected_profile = run(args.pixal3d, "--pipeline", "512")
    require(rejected_profile.returncode == 2, "Pixal3D accepted non-cascade pipeline", rejected_profile.stdout)
    require("must be 1024_cascade or 1536_cascade" in rejected_profile.stdout,
            "Pixal3D profile error is unclear", rejected_profile.stdout)

    rejected_trellis_profile = run(args.trellis2, "--pipeline", "1024_cascade")
    require(rejected_trellis_profile.returncode == 2,
            "TRELLIS.2 accepted a cascade pipeline", rejected_trellis_profile.stdout)
    require("must be 512 or 1024" in rejected_trellis_profile.stdout,
            "TRELLIS.2 profile error is unclear", rejected_trellis_profile.stdout)

    common = ("--dino", "unused-dino", "--image", "unused.png")
    wrong_trellis_family = run(
        args.trellis2,
        "--model",
        str(args.source_dir / "models" / "pixal3d"),
        "--pipeline",
        "1024",
        "--shape-only",
        "--prepared-image-output",
        "must-not-be-written.png",
        "--shape-latent-output",
        "must-not-be-written.tslat",
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
        "--shape-only",
        "--prepared-image-output",
        "must-not-be-written.png",
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
    for required_flag in ("--input", "--image", "--image-prepared", "--shape-latent", "--shape-latent-output", "--shape-encoder", "--texture-flow"):
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
        "--image-prepared",
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

    segmentation_help = run(args.segmentation, "--help")
    require(segmentation_help.returncode == 0,
            "TRELLIS.2 mesh segmentation --help failed", segmentation_help.stdout)
    require("Runs SegviGen Full inside the TRELLIS.2 runtime" in segmentation_help.stdout,
            "wrong TRELLIS.2 mesh segmentation task banner", segmentation_help.stdout)
    for required_flag in (
        "--segmentation-model", "--input", "--condition-image",
        "--rendered-condition", "--shape-latent", "--texture-latent",
        "--segmentation-latent",
        "--segmentation-flow", "--segmentation-latent-output",
        "--min-component-faces", "--small-part-mode",
    ):
        require(required_flag in segmentation_help.stdout,
                f"TRELLIS.2 mesh segmentation is missing {required_flag}",
                segmentation_help.stdout)
    for unrelated_flag in ("--naf", "--num-beams", "--texture-size"):
        require(unrelated_flag not in segmentation_help.stdout,
                f"TRELLIS.2 mesh segmentation exposes {unrelated_flag}",
                segmentation_help.stdout)

    segmentation_required = (
        "--model", "unused-base-model",
        "--segmentation-model", "unused-segmentation-model",
        "--dino", "unused-dino",
        "--input", "unused.glb",
        "--output", "unused-output.glb",
    )
    for option, value in (
        ("--steps", "0"),
        ("--steps", "1001"),
        ("--device", "-1"),
        ("--min-component-faces", "-1"),
        ("--min-palette-voxels", "0"),
        ("--palette-merge-distance", "-0.01"),
        ("--palette-merge-distance", "1.8"),
        ("--small-part-mode", "invalid"),
    ):
        rejected_range = run(
            args.segmentation,
            *segmentation_required,
            option,
            value,
        )
        require(rejected_range.returncode == 2,
                f"mesh segmentation accepted invalid {option}={value}",
                rejected_range.stdout)
        if option == "--small-part-mode":
            require("Usage:" in rejected_range.stdout,
                    "mesh segmentation did not reject invalid small-part mode",
                    rejected_range.stdout)
        else:
            require("invalid argument range" in rejected_range.stdout,
                    f"mesh segmentation did not explain invalid {option}",
                    rejected_range.stdout)

    for mode in ("keep", "merge", "discard"):
        accepted_mode = run(
            args.segmentation,
            *segmentation_required,
            "--small-part-mode",
            mode,
        )
        require(accepted_mode.returncode == 1,
                f"mesh segmentation did not parse small-part mode {mode}",
                accepted_mode.stdout)

    wrong_segmentation_task = run(
        args.segmentation,
        "--model",
        str(args.source_dir / "models" / "trellis2"),
        "--segmentation-model",
        str(args.source_dir / "models" / "trellis2"),
        "--dino",
        "must-not-be-opened-dino",
        "--input",
        "must-not-be-opened.glb",
        "--output",
        "must-not-be-written.glb",
    )
    require(wrong_segmentation_task.returncode == 1,
            "mesh segmentation accepted a non-segmentation package",
            wrong_segmentation_task.stdout)
    require("incompatible package contract" in wrong_segmentation_task.stdout,
            "mesh segmentation package error is unclear",
            wrong_segmentation_task.stdout)
    require("ggml_cuda_init" not in wrong_segmentation_task.stdout and
            "ggml_vulkan" not in wrong_segmentation_task.stdout and
            "failed to load mesh" not in wrong_segmentation_task.stdout,
            "mesh segmentation parsed input or initialized a GPU before package rejection",
            wrong_segmentation_task.stdout)

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_root = Path(temporary_directory)
        base_model = args.source_dir / "models" / "trellis2"
        segmentation_model = args.source_dir / "models" / "segvigen"
        input_path = temporary_root / "input.glb"
        input_path.write_bytes(b"preflight fixture")
        condition_path = temporary_root / "condition.png"
        condition_path.write_bytes(b"preflight fixture")
        cache_path = temporary_root / "cache.tslat"
        cache_path.write_bytes(b"preflight fixture")

        invalid_policy_dir = temporary_root / "invalid-policy"
        invalid_policy_dir.mkdir()
        invalid_manifest = json.loads(
            (segmentation_model / "model.json").read_text()
        )
        invalid_manifest["components"][0]["execution"][
            "emulate_bf16_blocks"
        ] = False
        (invalid_policy_dir / "model.json").write_text(
            json.dumps(invalid_manifest)
        )
        wrong_policy = run(
            args.segmentation,
            "--model", str(base_model),
            "--segmentation-model", str(invalid_policy_dir),
            "--dino", str(temporary_root / "missing-dino"),
            "--input", str(input_path),
            "--output", str(temporary_root / "policy-output.glb"),
        )
        require(wrong_policy.returncode == 1,
                "mesh segmentation accepted a non-official BF16 policy",
                wrong_policy.stdout)
        require(
            "compute_dtype=bf16 attention=flash flash_kv_dtype=bf16 "
            "emulate_bf16_blocks=true" in wrong_policy.stdout,
            "mesh segmentation policy error is unclear",
            wrong_policy.stdout,
        )
        require("ggml_cuda_init" not in wrong_policy.stdout and
                "ggml_vulkan" not in wrong_policy.stdout,
                "mesh segmentation initialized a GPU before policy rejection",
                wrong_policy.stdout)

        common_preflight = (
            "--model", str(base_model),
            "--segmentation-model", str(segmentation_model),
            "--dino", str(temporary_root / "missing-dino"),
            "--input", str(input_path),
        )
        input_output_conflict = run(
            args.segmentation,
            *common_preflight,
            "--output", str(input_path),
        )
        require(input_output_conflict.returncode == 1 and
                "path conflict" in input_output_conflict.stdout,
                "mesh segmentation did not reject input/output aliasing",
                input_output_conflict.stdout)

        condition_output_conflict = run(
            args.segmentation,
            *common_preflight,
            "--output", str(temporary_root / "parts.glb"),
            "--condition-image", str(condition_path),
            "--condition-prepared",
            "--rendered-condition", str(condition_path),
        )
        require(condition_output_conflict.returncode == 1 and
                "path conflict" in condition_output_conflict.stdout,
                "mesh segmentation did not reject condition/output aliasing",
                condition_output_conflict.stdout)

        cache_output_conflict = run(
            args.segmentation,
            *common_preflight,
            "--output", str(temporary_root / "parts.glb"),
            "--shape-latent", str(cache_path),
            "--texture-latent-output", str(cache_path),
        )
        require(cache_output_conflict.returncode == 1 and
                "path conflict" in cache_output_conflict.stdout,
                "mesh segmentation did not reject cache input/output aliasing",
                cache_output_conflict.stdout)

        cached_postprocess_without_dino = run(
            args.segmentation,
            "--model", str(base_model),
            "--segmentation-model", str(segmentation_model),
            "--input", str(input_path),
            "--output", str(temporary_root / "cached-output.glb"),
            "--segmentation-latent", str(cache_path),
        )
        require(cached_postprocess_without_dino.returncode == 1 and
                "segmentation latent cache" in cached_postprocess_without_dino.stdout,
                "cached postprocess incorrectly required --dino",
                cached_postprocess_without_dino.stdout)

        wrong_extension = run(
            args.segmentation,
            *common_preflight,
            "--output", str(temporary_root / "parts.gltf"),
        )
        require(wrong_extension.returncode == 1 and
                "must use the .glb extension" in wrong_extension.stdout,
                "mesh segmentation accepted a non-GLB output",
                wrong_extension.stdout)

        parent_blocker = temporary_root / "not-a-directory"
        parent_blocker.write_text("blocker")
        bad_parent = run(
            args.segmentation,
            *common_preflight,
            "--output", str(parent_blocker / "parts.glb"),
        )
        require(bad_parent.returncode == 1 and
                "parent directory" in bad_parent.stdout,
                "mesh segmentation did not reject an unusable output parent",
                bad_parent.stdout)

        nested_output = temporary_root / "created" / "early" / "parts.glb"
        missing_weight = run(
            args.segmentation,
            *common_preflight,
            "--output", str(nested_output),
        )
        require(missing_weight.returncode == 1 and
                "required segmentation flow weight" in missing_weight.stdout,
                "mesh segmentation did not preflight required model weights",
                missing_weight.stdout)
        require(nested_output.parent.is_dir(),
                "mesh segmentation did not prepare the output parent early",
                missing_weight.stdout)
        require("failed to load mesh" not in missing_weight.stdout and
                "ggml_cuda_init" not in missing_weight.stdout and
                "ggml_vulkan" not in missing_weight.stdout,
                "mesh segmentation parsed input or initialized a GPU before weight preflight",
                missing_weight.stdout)

        dummy_weight = temporary_root / "dummy.safetensors"
        dummy_weight.write_bytes(b"preflight fixture")
        missing_dino = run(
            args.segmentation,
            *common_preflight,
            "--output", str(temporary_root / "dino-output.glb"),
            "--segmentation-flow", str(dummy_weight),
            "--shape-encoder", str(dummy_weight),
            "--texture-encoder", str(dummy_weight),
            "--shape-decoder", str(dummy_weight),
            "--texture-decoder", str(dummy_weight),
        )
        require(missing_dino.returncode == 1 and
                "required DINO model weight" in missing_dino.stdout,
                "mesh segmentation did not preflight the DINO checkpoint",
                missing_dino.stdout)
        require("failed to load mesh" not in missing_dino.stdout and
                "ggml_cuda_init" not in missing_dino.stdout and
                "ggml_vulkan" not in missing_dino.stdout,
                "mesh segmentation reached input/GPU work before DINO preflight",
                missing_dino.stdout)

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_root = Path(temporary_directory)
        triangle_path = temporary_root / "triangle.gltf"
        triangle_path.write_text(json.dumps({
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": [{
                "attributes": {"POSITION": 0},
                "indices": 1,
                "mode": 4,
            }]}],
            "buffers": [{
                "uri": (
                    "data:application/octet-stream;base64,"
                    "AAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAAAAAAAAgD8AAAAAAAAAAAAA"
                ),
                "byteLength": 42,
            }],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962},
                {"buffer": 0, "byteOffset": 36, "byteLength": 6, "target": 34963},
            ],
            "accessors": [
                {
                    "bufferView": 0,
                    "componentType": 5126,
                    "count": 3,
                    "type": "VEC3",
                    "min": [0, 0, 0],
                    "max": [1, 1, 0],
                },
                {
                    "bufferView": 1,
                    "componentType": 5123,
                    "count": 3,
                    "type": "SCALAR",
                    "min": [0],
                    "max": [0],
                },
            ],
        }))

        missing_shape_decoder_package = temporary_root / "missing-shape-decoder"
        missing_shape_decoder_package.mkdir()
        missing_shape_decoder_manifest = json.loads(
            (args.source_dir / "models" / "trellis2" / "model.json").read_text()
        )
        missing_shape_decoder_manifest["components"] = [
            component
            for component in missing_shape_decoder_manifest["components"]
            if component.get("role") != "shape_decoder"
        ]
        (missing_shape_decoder_package / "model.json").write_text(
            json.dumps(missing_shape_decoder_manifest)
        )
        missing_shape_decoder = run(
            args.segmentation,
            "--model",
            str(missing_shape_decoder_package),
            "--segmentation-model",
            str(args.source_dir / "models" / "segvigen"),
            "--dino",
            "must-not-be-opened-dino",
            "--input",
            "must-not-be-opened.glb",
            "--output",
            str(temporary_root / "must-not-be-written-parts.glb"),
        )
        require(missing_shape_decoder.returncode == 1,
                "uncached segmentation accepted a package without shape_decoder",
                missing_shape_decoder.stdout)
        require("shape_decoder is always required" in missing_shape_decoder.stdout,
                "segmentation shape-decoder preflight error is unclear",
                missing_shape_decoder.stdout)
        require("ggml_cuda_init" not in missing_shape_decoder.stdout and
                "ggml_vulkan" not in missing_shape_decoder.stdout and
                "failed to load mesh" not in missing_shape_decoder.stdout,
                "shape-decoder preflight parsed input or initialized a GPU",
                missing_shape_decoder.stdout)

        prepared_texturing = run(
            args.texturing,
            "--model",
            str(args.source_dir / "models" / "trellis2"),
            "--dino",
            "must-not-be-opened-dino",
            "--input",
            str(triangle_path),
            "--image",
            "already-prepared.png",
            "--image-prepared",
            "--output",
            str(temporary_root / "must-not-be-written.glb"),
        )
        require(prepared_texturing.returncode == 1,
                "prepared-image texturing did not fail through the normal CLI path",
                prepared_texturing.stdout)
        require("trellis2-texture-mesh failed:" in prepared_texturing.stdout,
                "prepared-image texturing terminated before CLI error handling",
                prepared_texturing.stdout)
        require("ggml_cuda_init" not in prepared_texturing.stdout and
                "ggml_vulkan" not in prepared_texturing.stdout,
                "prepared-image timer regression initialized a GPU",
                prepared_texturing.stdout)

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
        missing_input = Path(temporary_directory) / "missing.gltf"
        valid_tokenskin_preflight = run(
            args.tokenskin,
            "--model",
            str(args.source_dir / "models" / "tokenskin"),
            "--input",
            str(missing_input),
            "--output",
            str(Path(temporary_directory) / "must-not-be-written.glb"),
        )
        require(valid_tokenskin_preflight.returncode == 1,
                "TokenSkin input preflight did not fail through the normal CLI path",
                valid_tokenskin_preflight.stdout)
        require("tokenskin-rig failed:" in valid_tokenskin_preflight.stdout,
                "TokenSkin terminated before CLI error handling",
                valid_tokenskin_preflight.stdout)
        require("TokenSkin input mesh:" in valid_tokenskin_preflight.stdout,
                "TokenSkin did not reach its expected input preflight",
                valid_tokenskin_preflight.stdout)

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
