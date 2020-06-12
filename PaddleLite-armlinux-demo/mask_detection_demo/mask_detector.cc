//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mask_detector.h"
#include <cstring>

using namespace paddle::lite_api;  // NOLINT

int64_t ShapeProduction(const std::vector<int64_t> &shape) {
  int64_t res = 1;
  for (auto i : shape)
    res *= i;
  return res;
}

void NHWC2NCHW(const float *src, float *dst, const float *mean,
               const float *std, int width, int height) {
  int size = height * width;
  float32x4_t vmean0 = vdupq_n_f32(mean ? mean[0] : 0.0f);
  float32x4_t vmean1 = vdupq_n_f32(mean ? mean[1] : 0.0f);
  float32x4_t vmean2 = vdupq_n_f32(mean ? mean[2] : 0.0f);
  float32x4_t vscale0 = vdupq_n_f32(std ? (1.0f / std[0]) : 1.0f);
  float32x4_t vscale1 = vdupq_n_f32(std ? (1.0f / std[1]) : 1.0f);
  float32x4_t vscale2 = vdupq_n_f32(std ? (1.0f / std[2]) : 1.0f);
  float *dst_c0 = dst;
  float *dst_c1 = dst + size;
  float *dst_c2 = dst + size * 2;
  int i = 0;
  for (; i < size - 3; i += 4) {
    float32x4x3_t vin3 = vld3q_f32(src);
    float32x4_t vsub0 = vsubq_f32(vin3.val[0], vmean0);
    float32x4_t vsub1 = vsubq_f32(vin3.val[1], vmean1);
    float32x4_t vsub2 = vsubq_f32(vin3.val[2], vmean2);
    float32x4_t vs0 = vmulq_f32(vsub0, vscale0);
    float32x4_t vs1 = vmulq_f32(vsub1, vscale1);
    float32x4_t vs2 = vmulq_f32(vsub2, vscale2);
    vst1q_f32(dst_c0, vs0);
    vst1q_f32(dst_c1, vs1);
    vst1q_f32(dst_c2, vs2);
    src += 12;
    dst_c0 += 4;
    dst_c1 += 4;
    dst_c2 += 4;
  }
  for (; i < size; i++) {
    *(dst_c0++) = (*(src++) - mean[0]) / std[0];
    *(dst_c1++) = (*(src++) - mean[1]) / std[1];
    *(dst_c2++) = (*(src++) - mean[2]) / std[2];
  }
}

FaceDetector::FaceDetector(const std::string &modelDir,
                           float inputScale,
                           const std::vector<float> &inputMean,
                           const std::vector<float> &inputStd,
                           float scoreThreshold)
    : inputScale_(inputScale),
      inputMean_(inputMean),
      inputStd_(inputStd),
      scoreThreshold_(scoreThreshold) {
  MobileConfig config;
  config.set_model_from_file(modelDir);
  predictor_ = CreatePaddlePredictor<MobileConfig>(config);
}

void FaceDetector::Preprocess(const cv::Mat &rgbaImage) {
  cv::Mat resizedRGBAImage;
  cv::resize(rgbaImage, resizedRGBAImage, cv::Size(), inputScale_, inputScale_,
             cv::INTER_CUBIC);
  cv::Mat resizedBGRImage;
  cv::cvtColor(resizedRGBAImage, resizedBGRImage, cv::COLOR_RGBA2BGR);
  resizedBGRImage.convertTo(resizedBGRImage, CV_32FC3, 1.0 / 255.0f);
  std::vector<int64_t> inputShape = {1, 3, resizedBGRImage.rows,
                                     resizedBGRImage.cols};
  // Prepare input tensor
  auto inputTensor = predictor_->GetInput(0);
  inputTensor->Resize(inputShape);
  auto inputData = inputTensor->mutable_data<float>();
  NHWC2NCHW(reinterpret_cast<const float *>(resizedBGRImage.data), inputData,
            inputMean_.data(), inputStd_.data(), inputShape[3], inputShape[2]);
}

void FaceDetector::Postprocess(const cv::Mat &rgbaImage,
                               std::vector<Face> *faces) {
  int imageWidth = rgbaImage.cols;
  int imageHeight = rgbaImage.rows;
  // Get output tensor
  auto outputTensor = predictor_->GetOutput(2);
  auto outputData = outputTensor->data<float>();
  auto outputShape = outputTensor->shape();
  int outputSize = ShapeProduction(outputShape);
  faces->clear();
  for (int i = 0; i < outputSize; i += 6) {
    // Class id
    float class_id = outputData[i];
    // Confidence score
    float score = outputData[i + 1];
    int left = outputData[i + 2] * imageWidth;
    int top = outputData[i + 3] * imageHeight;
    int right = outputData[i + 4] * imageWidth;
    int bottom = outputData[i + 5] * imageHeight;
    int width = right - left;
    int height = bottom - top;
    if (score > scoreThreshold_) {
      Face face;
      face.roi = cv::Rect(left, top, width, height) &
                 cv::Rect(0, 0, imageWidth - 1, imageHeight - 1);
      faces->push_back(face);
    }
  }
}

void FaceDetector::Predict(const cv::Mat &rgbaImage, std::vector<Face> *faces) {
  Preprocess(rgbaImage);
  predictor_->Run();
  Postprocess(rgbaImage, faces);
}

MaskClassifier::MaskClassifier(const std::string &modelDir,
                               int inputWidth,
                               int inputHeight,
                               const std::vector<float> &inputMean,
                               const std::vector<float> &inputStd)
    : inputWidth_(inputWidth), inputHeight_(inputHeight), inputMean_(inputMean),
      inputStd_(inputStd) {
  MobileConfig config;
  config.set_model_from_file(modelDir);
  predictor_ = CreatePaddlePredictor<MobileConfig>(config);
}

void MaskClassifier::Preprocess(const cv::Mat &rgbaImage,
                                const std::vector<Face> &faces) {
  // Prepare input tensor
  auto inputTensor = predictor_->GetInput(0);
  int batchSize = faces.size();
  std::vector<int64_t> inputShape = {batchSize, 3, inputHeight_, inputWidth_};
  inputTensor->Resize(inputShape);
  auto inputData = inputTensor->mutable_data<float>();
  for (int i = 0; i < batchSize; i++) {
    // Adjust the face region to improve the accuracy according to the aspect
    // ratio of input image of the target model
    int cx = faces[i].roi.x + faces[i].roi.width / 2.0f;
    int cy = faces[i].roi.y + faces[i].roi.height / 2.0f;
    int w = faces[i].roi.width;
    int h = faces[i].roi.height;
    float roiAspectRatio =
        static_cast<float>(faces[i].roi.width) / faces[i].roi.height;
    float inputAspectRatio = static_cast<float>(inputShape[3]) / inputShape[2];
    if (fabs(roiAspectRatio - inputAspectRatio) > 1e-5) {
      float widthRatio = static_cast<float>(faces[i].roi.width) / inputShape[3];
      float heightRatio =
          static_cast<float>(faces[i].roi.height) / inputShape[2];
      if (widthRatio > heightRatio) {
        h = w / inputAspectRatio;
      } else {
        w = h * inputAspectRatio;
      }
    }
    cv::Mat resizedRGBAImage(
        rgbaImage, cv::Rect(cx - w / 2, cy - h / 2, w, h) &
                       cv::Rect(0, 0, rgbaImage.cols - 1, rgbaImage.rows - 1));
    cv::resize(resizedRGBAImage, resizedRGBAImage,
               cv::Size(inputShape[3], inputShape[2]), 0.0f, 0.0f,
               cv::INTER_CUBIC);
    cv::Mat resizedBGRImage;
    cv::cvtColor(resizedRGBAImage, resizedBGRImage, cv::COLOR_RGBA2BGR);
    resizedBGRImage.convertTo(resizedBGRImage, CV_32FC3, 1.0 / 255.0f);
    NHWC2NCHW(reinterpret_cast<const float *>(resizedBGRImage.data), inputData,
              inputMean_.data(), inputStd_.data(), inputShape[3],
              inputShape[2]);
    inputData += inputShape[1] * inputShape[2] * inputShape[3];
  }
}

void MaskClassifier::Postprocess(std::vector<Face> *faces) {
  auto outputTensor = predictor_->GetOutput(0);
  auto outputData = outputTensor->data<float>();
  auto outputShape = outputTensor->shape();
  int outputSize = ShapeProduction(outputShape);
  int batchSize = faces->size();
  int classNum = outputSize / batchSize;
  for (int i = 0; i < batchSize; i++) {
    (*faces)[i].classid = 0;
    (*faces)[i].confidence = *(outputData++);
    for (int j = 1; j < classNum; j++) {
      auto confidence = *(outputData++);
      if (confidence > (*faces)[i].confidence) {
        (*faces)[i].classid = j;
        (*faces)[i].confidence = confidence;
      }
    }
  }
}

void MaskClassifier::Predict(const cv::Mat &rgbaImage, std::vector<Face> *faces) {
  Preprocess(rgbaImage, *faces);
  predictor_->Run();
  Postprocess(faces);
}
