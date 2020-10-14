#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>

#include <fisheye_corrector/fisheye_corrector.h>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <string>

#define usleep(x) Sleep((float)x / 1000.0f)

int main(int argc, char **argv) {
  // Retrieve paths to images
  for (int i = 0; i < argc; ++i) {
    std::cout << "string " << i << " : " << argv[i] << std::endl;
  }
  std::string video_path = argv[1];

  cv::VideoCapture video(video_path);
  int nImages = video.get(cv::CAP_PROP_FRAME_COUNT);
  double fps = video.get(cv::CAP_PROP_FPS);

  float pixel_height = 0.003;
  float f_image_ = 316.6;

  std::string correction_table = argv[2];
  std::cout << "generate corrector" << std::endl;
  FisheyeCorrector corrector;

  std::cout << video.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
  std::cout << video.get(cv::CAP_PROP_FRAME_WIDTH) << std::endl;
  std::cout << pixel_height << std::endl;
  std::cout << f_image_ << std::endl;

  corrector = FisheyeCorrector(
      correction_table, video.get(cv::CAP_PROP_FRAME_HEIGHT),
      video.get(cv::CAP_PROP_FRAME_WIDTH), pixel_height, f_image_, 37.05, 45.2);
  corrector.setAxisDirection(0, 20, 0);  // 30,35,-7
  corrector.updateMap();
  std::cout << "w: " << corrector.getCorrectedSize().width << std::endl;
  std::cout << "h: " << corrector.getCorrectedSize().height << std::endl;
  cv::Mat K = corrector.getIntrinsicMatrix();
  std::cout << "K: " << K << std::endl;
  // corrector.setClipRegion(cv::Rect(
  //     cv::Point(0, 475), cv::Point(corrector.getCorrectedSize().width,
  //                                  corrector.getCorrectedSize().height -
  //                                  500)));

  for (int i = 0; i < nImages; ++i) {
    cv::Mat frame;
    video >> frame;
    cv::Mat target;
    corrector.correct(frame, target);
    cv::imshow("distort frame", frame);
    cv::imshow("undistort frame", target);
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(6) << i;
    std::string file_name = "/home/kang/Pictures/rear/" + ss.str() + ".jpg";
    std::cout << "\rWrite Image: " << file_name << std::flush;
    cv::imwrite(file_name, target);
    cv::waitKey(10);
  }

  return 0;
}
