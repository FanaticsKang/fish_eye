#include "fisheye_corrector.h"
#include <opencv2/features2d/features2d.hpp>
// #include <debug_utils/debug_utils.h>

void FisheyeCorrector::readDistortionList(std::string file_name) {
  std::cout << "read distortion list" << std::endl;
  std::ifstream file(file_name);
  if (!file.is_open()) {
    std::cout << "open file error";
    exit(-1);
  }

  distortion_list_.reserve(1035);
  float current_distortion;
  char skip;
  while (file >> current_distortion) {
    distortion_list_.push_back(current_distortion);
  }
  file.close();
}

void FisheyeCorrector::generateMap() {
  float angle_between_original_axis = radianToDegree(
      atan((sin(axis_horizontal_radian_) * sin(axis_horizontal_radian_) +
            (tan(axis_vertical_radian_) * cos(axis_horizontal_radian_)) *
                (tan(axis_vertical_radian_) * cos(axis_horizontal_radian_))) /
           cos(axis_horizontal_radian_)));

  float radius_in_fisheye =
      distortion_list_[angle_between_original_axis * 10] / pixelHeight_;
  f_image_ = angle_between_original_axis < 0.01
                 ? f_camera_
                 : radius_in_fisheye /
                       sin(degreeToRadian(angle_between_original_axis));
  std::cout << "tan(horizontal_range_radian_)" << tan(horizontal_range_radian_) << std::endl;
  std::cout << "f_image: " << f_image_ << std::endl;

  Width_ = tan(horizontal_range_radian_) * f_image_ * 2;
  Height_ = tan(vertical_range_radian_) * f_image_ * 2;
  std::cout << "width " << Width_ << " Height " << Height_ << std::endl;
  CenterX_ = (float)Width_ / 2.0f;
  CenterY_ = (float)Height_ / 2.0f;
  map_ = cv::Mat::ones(Height_, Width_, CV_32FC2) * (-1);

  float radian_between_original_axis =
      degreeToRadian(angle_between_original_axis);
  float trans_z = cos(radian_between_original_axis) * f_image_;
  float trans_x = tan(axis_horizontal_radian_) * trans_z;
  float trans_y = tan(axis_vertical_radian_) * trans_z;
  new_camera_plane_center = Eigen::Vector3f(trans_x, trans_y, trans_z);
  Eigen::Vector3f new_camera_axis = new_camera_plane_center.normalized();

  original_axis = Eigen::Vector3f(0, 0, 1);

  Eigen::Quaternion<float> quaternion;

  quaternion.setFromTwoVectors(original_axis, new_camera_axis);
  quaternion.normalize();

  Eigen::Quaternion<float> quaternion_axis(cos(axis_rotation_radian_ / 2), 0, 0,
                                           sin(axis_rotation_radian_ / 2));
  quaternion_axis.normalize();
  Eigen::Matrix3f R_fisheye_camera =
      quaternion.toRotationMatrix() * quaternion_axis.toRotationMatrix();

  T_camera_fisheye = Eigen::Matrix4f::Identity();
  T_camera_fisheye.block(0, 0, 3, 3) = R_fisheye_camera.transpose();

  for (int h = 0; h < Height_; h++) {
    for (int w = 0; w < Width_; w++) {
      // Transform the points in the corrected image to it's correct position
      Eigen::Vector3f point_camera(w - CenterX_, h - CenterY_, f_image_);

      Eigen::Vector3f point_fisheye = R_fisheye_camera * point_camera;

      float cos_value = original_axis.dot(point_fisheye.normalized());

      float degree = radianToDegree(acos(cos_value));
      if (degree > 100) continue;

      // Intepolation for more acurate radius in correction table.
      int position_floor = floor(degree * 10);
      int position_ceil = ceil(degree * 10);
      float radius_in_fisheye_floor = distortion_list_[position_floor];
      float radius_in_fisheye_ceil = distortion_list_[position_ceil];
      float radius_in_fisheye;
      if (radius_in_fisheye_ceil == radius_in_fisheye_floor)
        radius_in_fisheye = radius_in_fisheye_ceil;
      else
        radius_in_fisheye = radius_in_fisheye_floor +
                            (radius_in_fisheye_ceil - radius_in_fisheye_floor) *
                                ((degree * 10 - position_floor) /
                                 (position_ceil - position_floor));

      radius_in_fisheye = radius_in_fisheye / pixelHeight_;

      float distance_to_original_axies =
          sqrt(point_fisheye(0) * point_fisheye(0) +
               point_fisheye(1) * point_fisheye(1));
      float x =
          point_fisheye(0) * (radius_in_fisheye / distance_to_original_axies);
      float y =
          point_fisheye(1) * (radius_in_fisheye / distance_to_original_axies);
      map_.at<cv::Vec2f>(h, w) =
          cv::Vec2f(x + CenterX_fisheye_, y + CenterY_fisheye_);
    }
  }
  // log_file.flush(); log_file.close();
  original_map_.release();
  map_.copyTo(original_map_);
}

FisheyeCorrector::FisheyeCorrector(std::string correction_table_file,
                                   int input_height, int input_width,
                                   float pixelHeight, float f,
                                   float vertical_range, float horizontal_range)
    : pixelHeight_(pixelHeight),
      f_camera_(f),
      horizontal_range_radian_(degreeToRadian(horizontal_range)),
      vertical_range_radian_(degreeToRadian(vertical_range)) {
  size_scale_ = 1;
  CenterX_fisheye_ = input_width / 2.0f;
  CenterY_fisheye_ = input_height / 2.0f;
  readDistortionList(correction_table_file);
  std::cout << distortion_list_.size() << std::endl;
  if (f_camera_ < distortion_list_[1] / sin(degreeToRadian(0.1))) {
    std::cout
        << "focal length of camera is too small. Please check if it's correct."
        << std::endl;
    exit(-1);
  }

  axis_vertical_radian_ = 0;
  axis_horizontal_radian_ = 0;
  axis_rotation_radian_ = 0;

  clip_region_ = cv::Rect(0, 0, 0, 0);
  map_need_update = true;
  new_camera_plane_center = Eigen::Vector3f(0, 0, 0);
  camera_center = Eigen::Vector3f(0, 0, 0);
  original_axis = Eigen::Vector3f(0, 0, 0);

  transform_camera_to_originalplane_ = Eigen::Matrix4f();
  T_camera_fisheye = Eigen::Matrix4f::Identity();
}

void FisheyeCorrector::updateMap() {
  if (original_map_.empty()) generateMap();

  if (clip_region_.area() == 0) clip_region_ = cv::Rect(0, 0, Width_, Height_);
  std::cout << clip_region_ << std::endl;
  std::cout << original_map_.size() << std::endl;
  original_map_(clip_region_).copyTo(map_);
  // map_to_original_plane(clip_region_).copyTo(map_to_original_plane_clip);
  cv::resize(map_, map_, cv::Size(0, 0), size_scale_, size_scale_,
             cv::INTER_NEAREST);

  // cv::resize(map_to_original_plane_clip,map_to_original_plane_clip,cv::Size(0,0),
  // size_scale_, size_scale_, cv::INTER_NEAREST);

  map_need_update = false;
}

template <>
void FisheyeCorrector::mapToOriginalImage<cv::KeyPoint>(
    const std::vector<cv::KeyPoint>& points,
    std::vector<cv::KeyPoint>& points_in_fisheye) {
  std::vector<cv::KeyPoint> points_in_fisheye_temp;
  points_in_fisheye_temp.resize(points.size());
  int width = map_.cols;
  int height = map_.rows;
  for (int i = 0; i < points.size(); i++) {
    float h = points[i].pt.y;
    float w = points[i].pt.x;
    // Transform the points in the corrected image to it's correct position

    if (h >= height || w >= width || h < 0 || w < 0) {
      points_in_fisheye_temp[i] = points[i];
      points_in_fisheye_temp[i].pt.x = -1;
      points_in_fisheye_temp[i].pt.y = -1;
      continue;
    }

    float x = map_.at<cv::Vec2f>(h, w)(0);
    float y = map_.at<cv::Vec2f>(h, w)(1);
    // std::cout << "x " << x << "   y " << y << std::endl;
    // Add the map relationship of Point(h,w)
    points_in_fisheye_temp[i] = points[i];
    points_in_fisheye_temp[i].pt.x = x;
    points_in_fisheye_temp[i].pt.y = y;
  }
  points_in_fisheye.clear();
  points_in_fisheye.insert(points_in_fisheye.end(),
                           points_in_fisheye_temp.begin(),
                           points_in_fisheye_temp.end());
}

template <>
void FisheyeCorrector::mapFromCorrectedImageToCenterImagePlane<cv::KeyPoint>(
    const std::vector<cv::KeyPoint>& points,
    std::vector<cv::KeyPoint>& points_in_pinhole, float cx, float cy,
    float f_center_image) {
  std::vector<cv::KeyPoint> points_in_pinhole_temp;
  points_in_pinhole_temp.resize(points.size());
  double ratio = f_center_image / f_camera_;
  int width = map_to_original_plane_clip.cols;
  int height = map_to_original_plane_clip.rows;
  // std::cout << "f_center_image " << f_center_image << " f_camera " <<
  // f_camera_ << std::endl;
  // std::cout << "ratio " << ratio << std::endl;
  for (int i = 0; i < points.size(); i++) {
    float h = points[i].pt.y;
    float w = points[i].pt.x;
    if (h >= height || w >= width || h < 0 || w < 0) {
      points_in_pinhole_temp[i] = points[i];
      points_in_pinhole_temp[i].pt.x = -1;
      points_in_pinhole_temp[i].pt.y = -1;
      continue;
    }
    // Transform the points in the corrected image to it's correct position
    float x = map_to_original_plane_clip.at<cv::Vec2f>(h, w)(0);
    float y = map_to_original_plane_clip.at<cv::Vec2f>(h, w)(1);
    // std::cout << "xo " << x << "   y " << y << std::endl;
    // Add the map relationship of Point(h,w)
    points_in_pinhole_temp[i] = points[i];
    points_in_pinhole_temp[i].pt.x = x * ratio + cx;
    points_in_pinhole_temp[i].pt.y = -y * ratio + cy;
  }
  points_in_pinhole.clear();
  points_in_pinhole_temp.swap(points_in_pinhole);
}