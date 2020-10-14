#pragma once
#include <math.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <vector>

const float kPI = 3.14159;
inline float degreeToRadian(float d) { return (d / 180.f) * kPI; }
inline float radianToDegree(float r) { return (r / kPI) * 180.f; }

class FisheyeCorrector {
  std::vector<float> distortion_list_;
  cv::Mat K_;

  cv::Mat original_map_;
  cv::Mat map_;
  cv::Mat map_to_original_plane;
  cv::Mat map_to_original_plane_clip;

  float f_camera_;
  float CenterX_fisheye_;
  float CenterY_fisheye_;
  float pixelHeight_;

  float vertical_range_radian_;
  float horizontal_range_radian_;
  int Width_;
  int Height_;
  float CenterX_;
  float CenterY_;

  float axis_vertical_radian_;
  float axis_horizontal_radian_;
  float axis_rotation_radian_;
  float f_image_;

  float size_scale_;
  Eigen::Matrix4f transform_camera_to_originalplane_;
  Eigen::Matrix4f T_camera_fisheye;
  cv::Rect clip_region_;

  Eigen::Vector3f new_camera_plane_center;
  Eigen::Vector3f camera_center;
  Eigen::Vector3f original_axis;

 private:
  void readDistortionList(std::string file_name);

  void generateMap();

  bool map_need_update = true;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // Correction table and pixelHeight should provided by camera manufactor.
  // focal length will infuence the size of the result image.
  FisheyeCorrector(std::string correction_table_file, int input_height,
                   int input_width, float pixelHeight, float f = 306.605,
                   float VerticalDegeree = 60, float HorizontalDegree = 70);
  FisheyeCorrector()
      : pixelHeight_(0),
        f_camera_(0),
        horizontal_range_radian_(degreeToRadian(0)),
        vertical_range_radian_(degreeToRadian(0)) {
    size_scale_ = 1;
    CenterX_fisheye_ = 0 / 2.0f;
    CenterY_fisheye_ = 0 / 2.0f;

    axis_vertical_radian_ = 0;
    axis_horizontal_radian_ = 0;
    axis_rotation_radian_ = 0;
    clip_region_ = cv::Rect(0, 0, 0, 0);
    map_need_update = true;
    new_camera_plane_center = Eigen::Vector3f(0, 0, 0);
    camera_center = Eigen::Vector3f(0, 0, 0);
    original_axis = Eigen::Vector3f(0, 0, 0);
    T_camera_fisheye = Eigen::Matrix4f::Identity();
  };

  FisheyeCorrector(const FisheyeCorrector& f) {
    distortion_list_.assign(f.distortion_list_.begin(),
                            f.distortion_list_.end());
    K_.release();
    f.K_.copyTo(K_);
    original_map_.release();
    f.original_map_.copyTo(original_map_);
    map_.release();
    map_.release();
    f.map_.copyTo(map_);
    f_camera_ = f.f_camera_;
    CenterX_fisheye_ = f.CenterX_fisheye_;
    CenterY_fisheye_ = f.CenterY_fisheye_;
    pixelHeight_ = f.pixelHeight_;

    vertical_range_radian_ = f.vertical_range_radian_;
    horizontal_range_radian_ = f.horizontal_range_radian_;
    Width_ = f.Width_;
    Height_ = f.Height_;
    CenterX_ = f.CenterX_;
    CenterY_ = f.CenterY_;

    axis_vertical_radian_ = f.axis_vertical_radian_;
    axis_horizontal_radian_ = f.axis_horizontal_radian_;
    axis_rotation_radian_ = f.axis_rotation_radian_;
    f_image_ = f.f_image_;

    size_scale_ = f.size_scale_;
    clip_region_ = f.clip_region_;
    transform_camera_to_originalplane_ = f.transform_camera_to_originalplane_;
    /*transform_camera_to_originalplane_ <<
       f.transform_camera_to_originalplane_(0, 0),
       f.transform_camera_to_originalplane_(0, 1),
       f.transform_camera_to_originalplane_(0, 2),
       f.transform_camera_to_originalplane_(0,3),
            f.transform_camera_to_originalplane_(1, 0),
       f.transform_camera_to_originalplane_(1, 1),
       f.transform_camera_to_originalplane_(1, 2),
       f.transform_camera_to_originalplane_(1, 3),
            f.transform_camera_to_originalplane_(2, 0),
       f.transform_camera_to_originalplane_(2, 1),
       f.transform_camera_to_originalplane_(2, 2),
       f.transform_camera_to_originalplane_(2, 3),
            f.transform_camera_to_originalplane_(3, 0),
       f.transform_camera_to_originalplane_(3, 1),
       f.transform_camera_to_originalplane_(3, 2),
       f.transform_camera_to_originalplane_(3, 3);*/
    //= Eigen::Matrix4f(f.transform_camera_to_originalplane_);
    new_camera_plane_center = Eigen::Vector3f(f.new_camera_plane_center);
    camera_center = Eigen::Vector3f(f.camera_center);
    original_axis = Eigen::Vector3f(f.original_axis);
    T_camera_fisheye = f.T_camera_fisheye;
  }

  FisheyeCorrector& operator=(const FisheyeCorrector& f) {
    if (this == &f) return *this;
    distortion_list_.assign(f.distortion_list_.begin(),
                            f.distortion_list_.end());
    K_.release();
    f.K_.copyTo(K_);
    original_map_.release();
    f.original_map_.copyTo(original_map_);
    map_.release();
    f.map_.copyTo(map_);
    f_camera_ = f.f_camera_;
    CenterX_fisheye_ = f.CenterX_fisheye_;
    CenterY_fisheye_ = f.CenterY_fisheye_;
    pixelHeight_ = f.pixelHeight_;

    vertical_range_radian_ = f.vertical_range_radian_;
    horizontal_range_radian_ = f.horizontal_range_radian_;
    Width_ = f.Width_;
    Height_ = f.Height_;
    CenterX_ = f.CenterX_;
    CenterY_ = f.CenterY_;

    axis_vertical_radian_ = f.axis_vertical_radian_;
    axis_horizontal_radian_ = f.axis_horizontal_radian_;
    axis_rotation_radian_ = f.axis_rotation_radian_;
    f_image_ = f.f_image_;

    size_scale_ = f.size_scale_;
    clip_region_ = f.clip_region_;

    transform_camera_to_originalplane_ = f.transform_camera_to_originalplane_;
    /*transform_camera_to_originalplane_ <<
       f.transform_camera_to_originalplane_(0, 0),
       f.transform_camera_to_originalplane_(0, 1),
       f.transform_camera_to_originalplane_(0, 2),
       f.transform_camera_to_originalplane_(0, 3),
            f.transform_camera_to_originalplane_(1, 0),
       f.transform_camera_to_originalplane_(1, 1),
       f.transform_camera_to_originalplane_(1, 2),
       f.transform_camera_to_originalplane_(1, 3),
            f.transform_camera_to_originalplane_(2, 0),
       f.transform_camera_to_originalplane_(2, 1),
       f.transform_camera_to_originalplane_(2, 2),
       f.transform_camera_to_originalplane_(2, 3),
            f.transform_camera_to_originalplane_(3, 0),
       f.transform_camera_to_originalplane_(3, 1),
       f.transform_camera_to_originalplane_(3, 2),
       f.transform_camera_to_originalplane_(3, 3);*/
    new_camera_plane_center = Eigen::Vector3f(f.new_camera_plane_center);
    camera_center = Eigen::Vector3f(f.camera_center);
    original_axis = Eigen::Vector3f(f.original_axis);
    T_camera_fisheye = f.T_camera_fisheye;
    return *this;
  }
  cv::Mat& correct(const cv::Mat& src, cv::Mat& dst) {
    if (map_need_update) updateMap();
    cv::remap(src, dst, map_, cv::Mat(), cv::INTER_CUBIC);
    /*		if (size_scale_!=1)
                            cv::resize(dst, dst, cv::Size(dst.cols*size_scale_,
       dst.rows*size_scale_), size_scale_, size_scale_, cv::INTER_CUBIC);*/
    return dst;
  }
  template <class pointType>
  void mapToOriginalImage(const std::vector<pointType>& points,
                          std::vector<pointType>& points_in_fisheye);

  template <class pointType>
  void mapFromCorrectedImageToCenterImagePlane(
      const std::vector<pointType>& points,
      std::vector<pointType>& points_in_pinhole, float cx, float cy,
      float f_center_image);

  cv::Size getCorrectedSize() {
    return cv::Size(original_map_.cols, original_map_.rows);
  }

  cv::Size getClipedSize() { return cv::Size(map_.cols, map_.rows); }

  void setClipRegion(const cv::Rect& region) {
    std::cout << region << std::endl;
    clip_region_ = region;
    map_ = original_map_(region);
    cv::resize(map_, map_, cv::Size(0, 0), size_scale_, size_scale_,
               cv::INTER_NEAREST);
    std::cout << "Size of corrected imageis  width:" << map_.cols
              << " height:" << map_.rows << std::endl;
  }

  void setSizeScale(float scale) { size_scale_ = scale; }

  cv::Mat getIntrinsicMatrix() {
    K_ = (cv::Mat_<float>(3, 3) << f_image_, 0, CenterX_ - clip_region_.x, 0,
          f_image_, CenterY_ - clip_region_.y, 0, 0, 1);
    K_ *= size_scale_;
    K_.at<float>(2, 2) /= size_scale_;
    return K_;
  }

  Eigen::Matrix4f TransformToFisheye() {
    return T_camera_fisheye.transpose();  // Only contain rotation component
  }

  Eigen::Matrix4f TransformToCamera() { return T_camera_fisheye; }

  void setAxisDirection(float axis_direction_horizontal,
                        float axis_direction_vertical, float axis_rotation) {
    axis_vertical_radian_ = -degreeToRadian(axis_direction_vertical);
    axis_horizontal_radian_ = degreeToRadian(axis_direction_horizontal);
    axis_rotation_radian_ = degreeToRadian(axis_rotation);
    map_need_update = true;
    generateMap();
  }
  void updateMap();
};

template <class pointType>
void FisheyeCorrector::mapToOriginalImage(
    const std::vector<pointType>& points,
    std::vector<pointType>& points_in_fisheye) {
  std::vector<pointType> points_in_fisheye_temp;
  points_in_fisheye_temp.resize(points.size());
  int width = original_map_.cols;
  int height = original_map_.rows;
  for (int i = 0; i < points.size(); i++) {
    float h = points[i].y / size_scale_ + clip_region_.tl().y;
    float w = points[i].x / size_scale_ + clip_region_.tl().x;
    // Transform the points in the corrected image to it's correct position

    if (h >= height || w >= width || h < 0 || w < 0) {
      points_in_fisheye_temp[i] = points[i];
      points_in_fisheye_temp[i].x = -1;
      points_in_fisheye_temp[i].y = -1;
      continue;
    }

    float x = original_map_.at<cv::Vec2f>(h, w)(0);
    float y = original_map_.at<cv::Vec2f>(h, w)(1);
    // std::cout << "x " << x << "   y " << y << std::endl;
    // Add the map relationship of Point(h,w)
    points_in_fisheye_temp[i] = points[i];
    points_in_fisheye_temp[i].x = x;
    points_in_fisheye_temp[i].y = y;
  }
  points_in_fisheye.clear();
  points_in_fisheye.insert(points_in_fisheye.end(),
                           points_in_fisheye_temp.begin(),
                           points_in_fisheye_temp.end());
}

template <class pointType>
void FisheyeCorrector::mapFromCorrectedImageToCenterImagePlane(
    const std::vector<pointType>& points,
    std::vector<pointType>& points_in_pinhole, float cx, float cy,
    float f_center_image) {
  std::vector<pointType> points_in_pinhole_temp;
  points_in_pinhole_temp.resize(points.size());
  double ratio = f_center_image / f_camera_;
  int width = map_to_original_plane.cols;
  int height = map_to_original_plane.rows;
  // std::cout << "f_center_image " << f_center_image << " f_camera " <<
  // f_camera_ << std::endl;
  // std::cout << "ratio " << ratio << std::endl;
  for (int i = 0; i < points.size(); i++) {
    float h = points[i].y / size_scale_ + clip_region_.tl().y;
    float w = points[i].x / size_scale_ + clip_region_.tl().x;
    if (h >= height || w >= width || h < 0 || w < 0) {
      points_in_pinhole_temp[i] = points[i];
      points_in_pinhole_temp[i].x = -1;
      points_in_pinhole_temp[i].y = -1;
      continue;
    }
    // Transform the points in the corrected image to it's correct position
    float x = map_to_original_plane.at<cv::Vec2f>(h, w)(0);
    float y = map_to_original_plane.at<cv::Vec2f>(h, w)(1);
    // std::cout << "xo " << x << "   y " << y << std::endl;
    // Add the map relationship of Point(h,w)
    points_in_pinhole_temp[i] = points[i];
    points_in_pinhole_temp[i].x = x * ratio + cx;
    points_in_pinhole_temp[i].y = -y * ratio + cy;
  }
  points_in_pinhole.clear();
  points_in_pinhole.insert(points_in_pinhole.end(),
                           points_in_pinhole_temp.begin(),
                           points_in_pinhole_temp.end());
}

template <>
void FisheyeCorrector::mapFromCorrectedImageToCenterImagePlane<cv::KeyPoint>(
    const std::vector<cv::KeyPoint>& points,
    std::vector<cv::KeyPoint>& points_in_pinhole, float cx, float cy,
    float f_center_image);
template <>
void FisheyeCorrector::mapToOriginalImage<cv::KeyPoint>(
    const std::vector<cv::KeyPoint>& points,
    std::vector<cv::KeyPoint>& points_in_fisheye);