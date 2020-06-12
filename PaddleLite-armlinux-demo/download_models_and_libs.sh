#!/bin/bash
set -e

# for classification demo
CLASSIFICATION_MODEL_DIR="$(pwd)/image_classification_demo/models/mobilenet_v1_for_cpu"
CLASSIFICATION_MODEL_URL="https://paddlelite-demo.bj.bcebos.com/models/mobilenet_v1_fp32_224_for_cpu_v2_6_0.tar.gz"

# for mask detetion
MASK_DETECTION_MODEL_0_DIR="$(pwd)/mask_detection_demo/models/pyramidbox_lite_for_cpu"
MASK_DETECTION_MODEL_0_URL="https://paddlelite-demo.bj.bcebos.com/models/pyramidbox_lite_fp32_for_cpu_v2_6_1.tar.gz"
MASK_DETECTION_MODEL_1_DIR="$(pwd)/mask_detection_demo/models/mask_detector_for_cpu"
MASK_DETECTION_MODEL_1_URL="https://paddlelite-demo.bj.bcebos.com/models/mask_detector_fp32_128_128_for_cpu_v2_6_1.tar.gz"

# for object detection demo
OBJECT_DETECTION_MODEL_DIR="$(pwd)/object_detection_demo/models/ssd_mobilenet_v1_pascalvoc_for_cpu"
OBJECT_DETECTION_MODEL_URL="https://paddlelite-demo.bj.bcebos.com/models/ssd_mobilenet_v1_pascalvoc_fp32_300_for_cpu_v2_6_0.tar.gz"

# paddlelite libs
LIBS_DIR="$(pwd)/Paddle-Lite"
LIBS_URL="https://paddlelite-demo.bj.bcebos.com/libs/armlinux/paddle_lite_libs_v2_6_0.tar.gz"

download_and_uncompress() {
  local url="$1"
  local dir="$2"
  
  echo "Start downloading ${url}"
  mkdir -p "${dir}"
  curl -L ${url} > ${dir}/download.tar.gz
  cd ${dir}
  tar -zxvf download.tar.gz
  rm -f download.tar.gz
}

download_and_uncompress "${OBJECT_DETECTION_MODEL_URL}" "${OBJECT_DETECTION_MODEL_DIR}"
download_and_uncompress "${MASK_DETECTION_MODEL_0_URL}" "${MASK_DETECTION_MODEL_0_DIR}"
download_and_uncompress "${MASK_DETECTION_MODEL_1_URL}" "${MASK_DETECTION_MODEL_1_DIR}"
download_and_uncompress "${CLASSIFICATION_MODEL_URL}" "${CLASSIFICATION_MODEL_DIR}"
download_and_uncompress "${LIBS_URL}" "${LIBS_DIR}"

echo "Download successful!"
