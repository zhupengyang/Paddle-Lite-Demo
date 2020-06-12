// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <iostream>
#include <string>
#include <vector>
#include "mask_detector.h"

using namespace paddle::lite_api;  // NOLINT

// Visualiztion MaskDetector results
void VisualizeResult(const cv::Mat& img,
                     const std::vector<Face>& results,
                     cv::Mat* vis_img) {
  for (int i = 0; i < results.size(); ++i) {
    cv::Rect roi = results[i].roi;

    // Configure color and text size
    cv::Scalar roi_color;
    std::string text;
    if (results[i].class_id == 1) {
      text = "MASK:  ";
      roi_color = cv::Scalar(0, 255, 0);
    } else {
      text = "NO MASK:  ";
      roi_color = cv::Scalar(0, 0, 255);
    }
    text += std::to_string(static_cast<int>(results[i].confidence * 100)) + "%";
    int font_face = cv::FONT_HERSHEY_TRIPLEX;
    double font_scale = 1.f;
    float thickness = 1;
    cv::Size text_size = cv::getTextSize(text,
                                         font_face,
                                         font_scale,
                                         thickness,
                                         nullptr);
    float new_font_scale = roi.width * font_scale / text_size.width;
    text_size = cv::getTextSize(text,
                               font_face,
                               new_font_scale,
                               thickness,
                               nullptr);
    cv::Point origin;
    origin.x = roi.x;
    origin.y = roi.y;

    // Configure text background
    cv::Rect text_back = cv::Rect(results[i].rect[0],
    results[i].rect[2] - text_size.height,
    text_size.width,
    text_size.height);

    // Draw roi object, text, and background
    *vis_img = img;
    cv::rectangle(*vis_img, roi, roi_color, 2);
    cv::rectangle(*vis_img, text_back, cv::Scalar(225, 225, 225), -1);
    cv::putText(*vis_img,
                text,
                origin,
                font_face,
                new_font_scale,
                cv::Scalar(0, 0, 0),
                thickness);
  }
}

int main(int argc, char* argv[]) {
  if (argc < 3 || argc > 4) {
    std::cout << "Usage:"
              << "./mask_detector ./models/ ./images/test.png"
              << std::endl;
    return -1;
  }

  auto det_model_dir = std::string(argv[1]);
  auto cls_model_dir = std::string(argv[2]);
  auto image_path = argv[3];

  // Init Detection Model
  float det_shrink = 0.6;
  float det_threshold = 0.7;
  std::vector<float> det_means = {104, 177, 123};
  std::vector<float> det_scale = {0.007843, 0.007843, 0.007843};
  FaceDetector detector(
      det_model_dir,
      det_shrink,
      det_means,
      det_scale,
      det_threshold);

  // Init Classification Model
  std::vector<float> cls_means = {0.5, 0.5, 0.5};
  std::vector<float> cls_scale = {1.0, 1.0, 1.0};
  MaskClassifier classifier(
      cls_model_dir,
      128,
      128,
      cls_means,
      cls_scale);

  // Load image
  cv::Mat img = imread(image_path, cv::IMREAD_COLOR);
  // Prediction result
  std::vector<Face> results;
  // Stage1: Face detection
  detector.Predict(img, &results, det_shrink);
  // Stage2: Mask wearing classification
  classifier.Predict(&results);

  // Visualization result
  cv::Mat vis_img;
  VisualizeResult(img, results, &vis_img);
  cv::imwrite("result.jpg", vis_img);

  return 0;
}