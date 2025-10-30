#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;

int main() {
  std::string image_path = "demo_photo.jpg";
  Mat img = imread(image_path, IMREAD_COLOR);

  if (img.empty()) {
    std::cout << "Could not read image: " << image_path << std::endl;
    return 1;
  }

  imshow("Display window", img);
  int k = waitKey(0); // wait for keystroke in windows
  return 0;
}
