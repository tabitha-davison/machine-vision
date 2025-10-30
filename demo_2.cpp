#include <iostream>
#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: ./text_detect <image_path>" << std::endl;
    return -1;
  }

  std::string image_path = argv[1];

  // Load image
  cv::Mat image = cv::imread(image_path);
  if (image.empty()) {
    std::cerr << "Error: Could not open or find the image." << std::endl;
    return -1;
  }

  // Convert to grayscale
  cv::Mat gray;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

  // Invert colors: white text becomes black on white background
  cv::Mat inverted;
  cv::bitwise_not(gray, inverted);

  // Initialize Tesseract OCR
  tesseract::TessBaseAPI ocr;
  if (ocr.Init(nullptr, "eng")) {
    std::cerr << "Could not initialize tesseract." << std::endl;
    return -1;
  }

  // Set page segmentation mode to single block for phone screens
  ocr.SetPageSegMode(
      tesseract::PSM_AUTO); // or PSM_SINGLE_BLOCK if text is in one block

  // Set the image
  ocr.SetImage(inverted.data, inverted.cols, inverted.rows, 1, inverted.step);
  ocr.Recognize(0);

  // Iterate through detected text lines
  tesseract::ResultIterator *ri = ocr.GetIterator();
  tesseract::PageIteratorLevel level = tesseract::RIL_TEXTLINE;

  if (ri != nullptr) {
    do {
      const char *text = ri->GetUTF8Text(level);
      float conf = ri->Confidence(level);
      int x1, y1, x2, y2;
      ri->BoundingBox(level, &x1, &y1, &x2, &y2);

      if (text && conf > 50) { // filter low-confidence text
        cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2),
                      cv::Scalar(0, 255, 0), 2);
        std::cout << "Detected text: \"" << text << "\" (conf=" << conf << ")"
                  << std::endl;
      }

      delete[] text;
    } while (ri->Next(level));
  }

  // Show and save the result
  cv::imshow("Detected Text", image);
  cv::imwrite("phone_text_result.png", image);
  cv::waitKey(0);

  ocr.End();
  return 0;
}
