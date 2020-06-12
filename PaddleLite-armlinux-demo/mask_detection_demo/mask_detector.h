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

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <utility>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "paddle_api.h" 

using namespace paddle::lite_api;  // NOLINT

// MaskDetector Result
struct Face {
  // Detection result: face rectangle
  cv::Rect roi;
  // Classification result: confidence
  float confidence;
  // Classification result : class id
  int classid;
};

class FaceDetector {
public:
  explicit FaceDetector(const std::string &modelDir, 
                        float inputScale,
                        const std::vector<float> &inputMean,
                        const std::vector<float> &inputStd,
                        float scoreThreshold);

  void Predict(const cv::Mat &rgbaImage, std::vector<Face> *faces);

private:
  void Preprocess(const cv::Mat &rgbaImage);
  void Postprocess(const cv::Mat &rgbaImage, std::vector<Face> *faces);

private:
  float inputScale_;
  std::vector<float> inputMean_;
  std::vector<float> inputStd_;
  float scoreThreshold_;
  std::shared_ptr<PaddlePredictor> predictor_;
};


class MaskClassifier {
public:
  explicit MaskClassifier(const std::string &modelDir,
                          int inputWidth,
                          int inputHeight,
                          const std::vector<float> &inputMean,
                          const std::vector<float> &inputStd);

  void Predict(const cv::Mat &rgbImage, std::vector<Face> *faces);

private:
  void Preprocess(const cv::Mat &rgbaImage, const std::vector<Face> &faces);
  void Postprocess(std::vector<Face> *faces);

private:
  int inputWidth_;
  int inputHeight_;
  std::vector<float> inputMean_;
  std::vector<float> inputStd_;
  std::shared_ptr<PaddlePredictor> predictor_;
};
