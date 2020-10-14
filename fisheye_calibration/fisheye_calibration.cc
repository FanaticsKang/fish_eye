#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fstream>
#include <iostream>

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

using std::string;

// 获取路径下的所有文件，linux版
bool GetFiles(const string& path, std::vector<string>* const files) {
  if (files == nullptr) {
    return false;
  }
  DIR* dir;
  struct dirent* ptr;
  if ((dir = opendir(path.c_str())) == NULL) {
    perror("Open dir error...");
    return false;
  }

  while ((ptr = readdir(dir)) != NULL) {
    if (strcmp(ptr->d_name, ".") == 0 ||
        strcmp(ptr->d_name, "..") == 0)  /// current dir OR parrent dir
      continue;
    else if (ptr->d_type == 8)  /// file
    {
      string strFile;
      strFile = path;
      strFile += "/";
      strFile += ptr->d_name;
      files->emplace_back(strFile);
    } else {
      continue;
    }
  }
  closedir(dir);
  return true;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout
        << "Input the filefold, like ./fish_eye_calibration /path/to/folder"
        << std::endl;
    return -1;
  }

  std::vector<string> files;

  // 获取该路径下的所有文件

  if (!GetFiles(argv[1], &files) || files.size() == 0) {
    std::cout << "No images" << std::endl;
    return -1;
  }

  const int board_w = 8;
  const int board_h = 6;
  const int NPoints = board_w * board_h;  //棋盘格内角点总数
  const int boardSize = 30;               // mm
  cv::Mat image, grayimage;
  cv::Size ChessBoardSize = cv::Size(board_w, board_h);

  int flag = 0;
  flag |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
  // flag |= cv::fisheye::CALIB_CHECK_COND;
  flag |= cv::fisheye::CALIB_FIX_SKEW;
  // flag |= cv::fisheye::CALIB_USE_INTRINSIC_GUESS;

  std::vector<cv::Point3f> object;
  for (int j = 0; j < NPoints; ++j) {
    //构造棋盘格坐标系
    object.emplace_back((j % board_w) * boardSize, (j / board_w) * boardSize,
                        0);
  }

  cv::Matx33d intrinsics;      // z:相机内参
  cv::Vec4d distortion_coeff;  // z:相机畸变系数

  std::vector<std::vector<cv::Point3f> > objectv;
  std::vector<std::vector<cv::Point2f> > imagev;

  cv::Size corrected_size(1280, 720);
  cv::Mat mapx, mapy;
  cv::Mat corrected;

  int has_checkboard = 0;
  std::cout << "capture image: ";

  for (size_t num = 0; num < files.size(); ++num) {
    image = cv::imread(files[num]);
    if (image.empty()) {
      break;
    }

    cv::imshow("corner_image", image);
    cv::waitKey(10);
    cvtColor(image, grayimage, CV_BGR2GRAY);
    IplImage tempgray = grayimage;
    bool findchessboard = cvCheckChessboard(&tempgray, ChessBoardSize);

    if (findchessboard) {
      std::vector<cv::Point2f> tempcorners;
      bool find_corners_result =
          findChessboardCorners(grayimage, ChessBoardSize, tempcorners, 3);
      if (find_corners_result) {
        cornerSubPix(
            grayimage, tempcorners, cvSize(5, 5), cvSize(-1, -1),
            cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
        drawChessboardCorners(image, ChessBoardSize, tempcorners,
                              find_corners_result);
        imshow("corner_image", image);
        cvWaitKey(100);

        std::cout << num << ", " << std::flush;
        objectv.emplace_back(object);
        imagev.emplace_back(tempcorners);
        ++has_checkboard;
      }
    }
  }
  std::cout << "\nSuccess/Total image: " << has_checkboard << "/"
            << files.size() << std::endl;

  cv::fisheye::calibrate(objectv, imagev, cv::Size(image.cols, image.rows),
                         intrinsics, distortion_coeff, cv::noArray(),
                         cv::noArray(), flag, cv::TermCriteria(3, 20, 1e-6));
  cv::fisheye::initUndistortRectifyMap(intrinsics, distortion_coeff,
                                       cv::Matx33d::eye(), intrinsics,
                                       corrected_size, CV_16SC2, mapx, mapy);
  cv::FileStorage cv_file("camera.yaml", cv::FileStorage::WRITE);

  cv_file << "Camera_type"
          << "KannalaBrandt8";

  cv_file << "Camera_fx" << intrinsics(0, 0);
  cv_file << "Camera_fy" << intrinsics(1, 1);
  cv_file << "Camera_cx" << intrinsics(0, 2);
  cv_file << "Camera_cy" << intrinsics(1, 2);

  std::cout << "camera intrisic: \n" << intrinsics << std::endl;

  cv_file << "Camera_k1" << distortion_coeff(0);
  cv_file << "Camera_k2" << distortion_coeff(1);
  cv_file << "Camera_k3" << distortion_coeff(2);
  cv_file << "Camera_k4" << distortion_coeff(3);
  std::cout << "distortion coeff: \n" << distortion_coeff << std::endl;

  cv_file << "Camera_width" << image.size[1];
  cv_file << "Camera_height" << image.size[0];
  std::cout << "image size: " << image.size[1] << " x " << image.size[0]
            << std::endl;

  // show corrected image
  // for (size_t num = 0; num < files.size(); ++num) {
  //   image = cv::imread(files[num++]);

  //   if (image.empty()) break;
  //   remap(image, corrected, mapx, mapy, cv::INTER_LINEAR,
  //         cv::BORDER_TRANSPARENT);

  //   cv::imshow("corner_image", image);
  //   cv::imshow("corrected", corrected);
  //   cvWaitKey(200);
  // }
  std::cout << "press ESC to quit." << std::endl;
  cv::waitKey(-1);

  image.release();
  grayimage.release();
  corrected.release();
  mapx.release();
  mapy.release();

  return 0;
}