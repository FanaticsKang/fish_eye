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
  GetFiles(argv[1], &files);

  const int board_w = 8;
  const int board_h = 6;
  const int NPoints = board_w * board_h;  //棋盘格内角点总数
  const int boardSize = 30;               // mm
  cv::Mat image, grayimage;
  cv::Size ChessBoardSize = cv::Size(board_w, board_h);
  std::vector<cv::Point2f> tempcorners;

  int flag = 0;
  flag |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
  // flag |= cv::fisheye::CALIB_CHECK_COND;
  flag |= cv::fisheye::CALIB_FIX_SKEW;
  // flag |= cv::fisheye::CALIB_USE_INTRINSIC_GUESS;

  std::vector<cv::Point3f> object;
  for (int j = 0; j < NPoints; ++j) {
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

  std::ofstream intrinsicfile("intrinsics_front1103.txt");
  std::ofstream disfile("dis_coeff_front1103.txt");
  int num = 0;
  bool bCalib = false;
  while (num < files.size()) {
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

        objectv.emplace_back(object);
        imagev.emplace_back(tempcorners);
        std::cout << "capture " << num << " pictures" << std::endl;
      }
    }
    tempcorners.clear();
    num++;
  }

  cv::fisheye::calibrate(objectv, imagev, cv::Size(image.cols, image.rows),
                         intrinsics, distortion_coeff, cv::noArray(),
                         cv::noArray(), flag, cv::TermCriteria(3, 20, 1e-6));
  cv::fisheye::initUndistortRectifyMap(intrinsics, distortion_coeff,
                                       cv::Matx33d::eye(), intrinsics,
                                       corrected_size, CV_16SC2, mapx, mapy);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      intrinsicfile << intrinsics(i, j) << "\t";
    }
    intrinsicfile << std::endl;
  }
  for (int i = 0; i < 4; ++i) {
    disfile << distortion_coeff(i) << "\t";
  }
  intrinsicfile.close();
  disfile.close();

  num = 0;
  while (num < files.size()) {
    image = cv::imread(files[num++]);

    if (image.empty()) break;
    remap(image, corrected, mapx, mapy, cv::INTER_LINEAR,
          cv::BORDER_TRANSPARENT);

    cv::imshow("corner_image", image);
    cv::imshow("corrected", corrected);
    cvWaitKey(200);
  }

  cv::destroyWindow("corner_image");
  cv::destroyWindow("corrected");

  image.release();
  grayimage.release();
  corrected.release();
  mapx.release();
  mapy.release();

  return 0;
}