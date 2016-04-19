#!/bin/bash
# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#  ./download_imagenet.sh [dirname]
set -e

OUTDIR="/home/tammy/imagenet-data/"
SYNSETS_FILE="/home/tammy/SOURCES/models/inception/inception/data/imagenet_2012_validation_synset_labels.txt"

CURRENT_DIR=$(pwd)
BBOX_DIR="${OUTDIR}bounding_boxes"
cd "${OUTDIR}"

# See here for details: http://www.image-net.org/download-bboxes
BBOX_TAR_BALL="${BBOX_DIR}/ILSVRC2012_bbox_train_v2.tar.gz" 
tar xzf "${BBOX_TAR_BALL}" -C "${BBOX_DIR}"

LABELS_ANNOTATED="${BBOX_DIR}/*"
NUM_XML=$(ls -1 ${LABELS_ANNOTATED} | wc -l)
echo "Identified ${NUM_XML} bounding box annotations."

# Download and uncompress all images from the ImageNet 2012 validation dataset.
# VALIDATION_TARBALL="${OUTDIR}ILSVRC2012_img_val.tar"
# OUTPUT_PATH="${OUTDIR}validation/"
# mkdir -p "${OUTPUT_PATH}"
# tar xf "${VALIDATION_TARBALL}" -C "${OUTPUT_PATH}"

# Download all images from the ImageNet 2012 train dataset.
# TRAIN_TARBALL="${OUTDIR}ILSVRC2012_img_train.tar"
OUTPUT_PATH="${OUTDIR}train/"
# Un-compress the individual tar-files within the train tar-file.
while read SYNSET; do
  # echo "Processing: ${SYNSET}"
  FILE="/home/tammy/imagenet-data/ILSVRC2012_img_train/${SYNSET}.tar"
  if [ -f $FILE ];
  then
    mkdir -p "${OUTPUT_PATH}/${SYNSET}"
    rm -rf "${OUTPUT_PATH}/${SYNSET}/*"

    tar xf $FILE -C "${OUTPUT_PATH}/${SYNSET}/"
  else
    echo "$FILE does not exist."
  fi
  # echo "Finished processing: ${SYNSET}"
done < "${SYNSETS_FILE}"
