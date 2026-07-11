#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DIST_DIR="$REPO_ROOT/dist"
COMMIT=""
VULKAN_BUILD="$REPO_ROOT/build-vulkan"
CUDA_BUILD="$REPO_ROOT/build-cuda"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '2')}"
SKIP_BUILD=0
MAKE_ARCHIVE=1
DRY_RUN=0
BUNDLE_SYSTEM_LIBS=0

usage() {
    cat <<'USAGE'
Usage:
  tools/package_linux_release.sh [options]

Build and package four Linux x64 runtime bundles:
  - trellis2-local-linux-x64-vulkan-gui
  - trellis2-local-linux-x64-vulkan-cli
  - trellis2-local-linux-x64-cuda-gui
  - trellis2-local-linux-x64-cuda-cli

Options:
  --dist-dir DIR          Output directory, default ./dist
  --commit NAME           Package suffix metadata, default git short SHA
  --vulkan-build DIR      Vulkan build directory, default ./build-vulkan
  --cuda-build DIR        CUDA build directory, default ./build-cuda
  --jobs N                Parallel build jobs
  --no-build              Do not invoke cmake --build
  --no-archive            Leave package directories only; skip .tar.gz creation
  --bundle-system-libs    Also copy non-glibc system libraries reported by ldd
  --dry-run               Check inputs and print the package plan without copying
  -h, --help              Show this help
USAGE
}

abs_from_repo() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s/%s\n' "$REPO_ROOT" "$1" ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dist-dir)
            DIST_DIR="$(abs_from_repo "$2")"
            shift 2
            ;;
        --commit)
            COMMIT="$2"
            shift 2
            ;;
        --vulkan-build)
            VULKAN_BUILD="$(abs_from_repo "$2")"
            shift 2
            ;;
        --cuda-build)
            CUDA_BUILD="$(abs_from_repo "$2")"
            shift 2
            ;;
        --jobs)
            JOBS="$2"
            shift 2
            ;;
        --no-build)
            SKIP_BUILD=1
            shift
            ;;
        --no-archive)
            MAKE_ARCHIVE=0
            shift
            ;;
        --bundle-system-libs)
            BUNDLE_SYSTEM_LIBS=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$COMMIT" ]]; then
    COMMIT="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || printf 'unknown')"
fi

ARCH="$(uname -m)"
case "$ARCH" in
    x86_64|amd64) ARCH_TAG="x64" ;;
    aarch64|arm64) ARCH_TAG="arm64" ;;
    *) ARCH_TAG="$ARCH" ;;
esac

ensure_dir() {
    if [[ "$DRY_RUN" -eq 0 ]]; then
        mkdir -p "$1"
    fi
}

reset_dir() {
    if [[ "$DRY_RUN" -eq 0 ]]; then
        rm -rf "$1"
        mkdir -p "$1"
    fi
}

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Missing required file: $1" >&2
        exit 1
    fi
}

require_executable() {
    if [[ ! -x "$1" ]]; then
        echo "Missing required executable: $1" >&2
        exit 1
    fi
}

copy_file() {
    local src="$1"
    local dst_dir="$2"
    require_file "$src"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        cp -f "$src" "$dst_dir/"
    fi
}

copy_exec() {
    local src="$1"
    local dst_dir="$2"
    require_executable "$src"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        cp -f "$src" "$dst_dir/"
        chmod 755 "$dst_dir/$(basename "$src")"
    fi
}

copy_library() {
    local lib="$1"
    local dst_dir="$2"
    [[ -f "$lib" || -L "$lib" ]] || return 0
    local real
    real="$(readlink -f "$lib")"
    [[ -f "$real" ]] || return 0
    local real_name
    real_name="$(basename "$real")"
    local soname
    soname="$(basename "$lib")"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        cp -f "$real" "$dst_dir/$real_name"
        chmod 644 "$dst_dir/$real_name"
        if [[ "$soname" != "$real_name" ]]; then
            ln -sf "$real_name" "$dst_dir/$soname"
        fi
    fi
}

ldd_paths() {
    local binary="$1"
    ldd "$binary" 2>/dev/null | awk '
        /=>/ {
            for (i = 1; i <= NF; ++i) {
                if ($i == "=>") {
                    print $(i + 1)
                }
            }
        }
        /^[[:space:]]*\// {
            print $1
        }
    ' | sed '/^not$/d;/^statically$/d;/^linux-vdso/d'
}

is_glibc_core() {
    case "$(basename "$1")" in
        ld-linux*.so*|libc.so.*|libm.so.*|libdl.so.*|libpthread.so.*|librt.so.*|libresolv.so.*|libnsl.so.*|libutil.so.*)
            return 0
            ;;
    esac
    return 1
}

is_runtime_lib_to_bundle() {
    local path="$1"
    local base
    base="$(basename "$path")"
    if [[ "$path" == "$REPO_ROOT/"* ]]; then
        return 0
    fi
    case "$base" in
        libcuda.so.*)
            return 1
            ;;
        libvulkan.so.*|libOpenCL.so.*|libgomp.so.*|libstdc++.so.*|libgcc_s.so.*)
            return 0
            ;;
        libcudart.so.*|libcublas.so.*|libcublasLt.so.*|libnvrtc.so.*|libnvJitLink.so.*|libcurand.so.*)
            return 0
            ;;
    esac
    if [[ "$BUNDLE_SYSTEM_LIBS" -eq 1 ]] && ! is_glibc_core "$path"; then
        return 0
    fi
    return 1
}

copy_runtime_libs() {
    local dst_dir="$1"
    shift
    local seen=""
    local binary lib
    for binary in "$@"; do
        [[ -x "$binary" ]] || continue
        while IFS= read -r lib; do
            [[ -n "$lib" && "$lib" != "not" && -e "$lib" ]] || continue
            if is_runtime_lib_to_bundle "$lib" && [[ ":$seen:" != *":$lib:"* ]]; then
                copy_library "$lib" "$dst_dir"
                seen="$seen:$lib"
            fi
        done < <(ldd_paths "$binary")
    done
}

configure_build_if_needed() {
    local build_dir="$1"
    local backend="$2"
    if [[ -f "$build_dir/CMakeCache.txt" ]]; then
        return
    fi
    cmake -S "$REPO_ROOT" -B "$build_dir" \
        -DTRELLIS2_C_BACKEND="$backend" \
        -DCMAKE_BUILD_TYPE=Release
}

build_targets() {
    if [[ "$SKIP_BUILD" -eq 1 || "$DRY_RUN" -eq 1 ]]; then
        return
    fi
    configure_build_if_needed "$VULKAN_BUILD" vulkan
    configure_build_if_needed "$CUDA_BUILD" cuda
    cmake --build "$VULKAN_BUILD" --target \
        trellis-gui trellis-image-to-gltf trellis-birefnet-rgba trellis-rebake-gltf vkmesh \
        -j "$JOBS"
    cmake --build "$CUDA_BUILD" --target \
        trellis-gui trellis-image-to-gltf trellis-birefnet-rgba trellis-rebake-gltf vkmesh \
        -j "$JOBS"
}

write_download_scripts() {
    local pkg="$1"
    ensure_dir "$pkg/tools"
    copy_file "$REPO_ROOT/tools/download_weights.py" "$pkg/tools"
    if [[ "$DRY_RUN" -ne 0 ]]; then
        return
    fi
    cat > "$pkg/download_weights.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$HERE/TRELLIS.2"
PYTHON_BIN="${PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN=python3
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN=python
    else
        echo "Python 3 was not found. Install Python 3, then run this script again." >&2
        exit 1
    fi
fi

VENV_DIR="$HERE/.download-venv"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR" >/dev/null 2>&1 || true
fi
if [[ -x "$VENV_DIR/bin/python" ]]; then
    PYTHON_BIN="$VENV_DIR/bin/python"
fi

"$PYTHON_BIN" -m pip install -U "huggingface_hub[cli]"
case " $* " in
    *" modelscope "*|*" ms "*|*"--source=modelscope"*|*"--source=ms"*)
        "$PYTHON_BIN" -m pip install -U modelscope
        ;;
esac

mkdir -p "$OUT_DIR"
"$PYTHON_BIN" "$HERE/tools/download_weights.py" --output-dir "$OUT_DIR" "$@"
echo
echo "Weights are ready in: $OUT_DIR"
SH
    chmod 755 "$pkg/download_weights.sh"
}

write_run_scripts() {
    local pkg="$1"
    local backend="$2"
    local flavor="$3"
    if [[ "$DRY_RUN" -ne 0 ]]; then
        return
    fi
    if [[ "$flavor" == "gui" ]]; then
        cat > "$pkg/run-gui.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="$HERE/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$HERE/bin:$PATH"
exec "$HERE/bin/trellis-gui" --weights "$HERE/TRELLIS.2" "$@"
SH
        chmod 755 "$pkg/run-gui.sh"
    else
        cat > "$pkg/run-cli.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="$HERE/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$HERE/bin:$PATH"
exec "$HERE/bin/trellis-image-to-gltf" \
    --model "$HERE/TRELLIS.2/TRELLIS.2-4B" \
    --dino "$HERE/TRELLIS.2/dinov3-vitl16-pretrain-lvd1689m" \
    "$@"
SH
        chmod 755 "$pkg/run-cli.sh"
    fi

    cat > "$pkg/env.sh" <<'SH'
#!/usr/bin/env bash
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="$HERE/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$HERE/bin:$PATH"
SH
    chmod 755 "$pkg/env.sh"

    cat > "$pkg/README.txt" <<EOF
trellis2 local Linux $ARCH_TAG $backend $flavor package
Commit: $COMMIT

Weights:
  ./download_weights.sh
  ./download_weights.sh --source modelscope

Run:
EOF
    if [[ "$flavor" == "gui" ]]; then
        cat >> "$pkg/README.txt" <<'EOF'
  ./run-gui.sh

The GUI writes outputs to viewer_outputs/ next to this package.
EOF
    else
        cat >> "$pkg/README.txt" <<'EOF'
  ./run-cli.sh --image images.jpg --output output.glb

You can pass any trellis-image-to-gltf option after ./run-cli.sh.
EOF
    fi
    cat >> "$pkg/README.txt" <<EOF

Included:
  - bin/ runtime executables
  - lib/ bundled project and backend runtime libraries
  - download_weights.sh plus tools/download_weights.py
  - images.jpg sample image when available

Notes:
  - GPU drivers are not bundled.
  - CUDA packages include CUDA runtime libraries found by ldd, except libcuda.so.1.
  - Vulkan packages include the Vulkan loader when found; an installed Vulkan driver is still required.
EOF
}

copy_example_image() {
    local pkg="$1"
    local image="$REPO_ROOT/example_image/images.jpg"
    if [[ -f "$image" ]]; then
        if [[ "$DRY_RUN" -eq 0 ]]; then
            cp -f "$image" "$pkg/images.jpg"
            mkdir -p "$pkg/example_image"
            cp -f "$image" "$pkg/example_image/images.jpg"
        fi
    fi
}

copy_common_tools() {
    local build_dir="$1"
    local pkg="$2"
    local flavor="$3"
    ensure_dir "$pkg/bin"
    ensure_dir "$pkg/lib"
    if [[ "$flavor" == "gui" ]]; then
        copy_exec "$build_dir/trellis-gui" "$pkg/bin"
    else
        copy_exec "$build_dir/trellis-image-to-gltf" "$pkg/bin"
        copy_exec "$build_dir/trellis-birefnet-rgba" "$pkg/bin"
        copy_exec "$build_dir/trellis-rebake-gltf" "$pkg/bin"
    fi
    copy_exec "$build_dir/vkmesh" "$pkg/bin"
    write_download_scripts "$pkg"
    copy_example_image "$pkg"
}

package_one() {
    local backend="$1"
    local flavor="$2"
    local build_dir="$3"
    local name="trellis2-local-linux-${ARCH_TAG}-${backend}-${flavor}"
    local pkg="$DIST_DIR/$name"
    echo "Package: $name"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        reset_dir "$pkg"
    fi
    copy_common_tools "$build_dir" "$pkg" "$flavor"

    local binaries=("$build_dir/vkmesh")
    if [[ "$flavor" == "gui" ]]; then
        binaries+=("$build_dir/trellis-gui")
    else
        binaries+=("$build_dir/trellis-image-to-gltf" "$build_dir/trellis-birefnet-rgba" "$build_dir/trellis-rebake-gltf")
    fi
    if [[ "$DRY_RUN" -eq 0 ]]; then
        copy_runtime_libs "$pkg/lib" "${binaries[@]}"
        write_run_scripts "$pkg" "$backend" "$flavor"
    fi

    if [[ "$MAKE_ARCHIVE" -eq 1 ]]; then
        local archive="$DIST_DIR/${name}-${COMMIT}.tar.gz"
        if [[ "$DRY_RUN" -eq 0 ]]; then
            rm -f "$archive"
            tar -C "$DIST_DIR" -czf "$archive" "$name"
        fi
        echo "  archive: $archive"
    else
        echo "  dir: $pkg"
    fi
}

build_targets

require_executable "$VULKAN_BUILD/vkmesh"
for target in trellis-gui trellis-image-to-gltf trellis-birefnet-rgba trellis-rebake-gltf; do
    require_executable "$VULKAN_BUILD/$target"
done
for target in trellis-gui trellis-image-to-gltf trellis-birefnet-rgba trellis-rebake-gltf vkmesh; do
    require_executable "$CUDA_BUILD/$target"
done

ensure_dir "$DIST_DIR"
package_one vulkan gui "$VULKAN_BUILD"
package_one vulkan cli "$VULKAN_BUILD"
package_one cuda gui "$CUDA_BUILD"
package_one cuda cli "$CUDA_BUILD"

echo "Done."
