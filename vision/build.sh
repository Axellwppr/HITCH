#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
for candidate in "${REALSENSE_PREFIX:-}" /usr/local /usr /opt/ros/*; do
  if [[ -n "$candidate" && -f "$candidate/include/librealsense2/rs.hpp" ]]; then
    prefix="$candidate"
    break
  fi
done
prefix="${prefix:-${REALSENSE_PREFIX:-/usr/local}}"
libdir="$prefix/lib/$(uname -m)-linux-gnu"
[[ -f "$libdir/librealsense2.so" ]] || libdir="$prefix/lib"
g++ -O2 -std=c++11 capture_rgbd.cpp -I"$prefix/include" -L"$libdir" -Wl,-rpath,"$libdir" -lrealsense2 -o build/capture_rgbd
