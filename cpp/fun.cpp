#include <iostream>
#include <opencv2/opencv.hpp>

int main() {

    auto image = cv::imread("../robert_de_niro.jpg");

    auto gray = cv::Mat();
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    auto edges = cv::Mat();
    cv::Sobel(gray, edges, CV_32F, 1, 0);
    edges = cv::abs(edges);
    cv::normalize(edges, edges, 0, 1, cv::NORM_MINMAX, CV_32F);
    cv::pow(edges, 0.25, edges);

    auto blurred = cv::Mat();
    cv::medianBlur(image, blurred, 21);

    auto final = cv::Mat();

    image.convertTo(image, CV_32FC3);
    blurred.convertTo(blurred, CV_32FC3);

    cv::Mat image_channels[3];
    cv::split(image, image_channels);
    cv::Mat blurred_channels[3];
    cv::split(blurred, blurred_channels);
    cv::Mat final_channels[3];

    for(int i = 0; i < 3; ++i)
        final_channels[i] = image_channels[i].mul(edges) + blurred_channels[i].mul(1 - edges);
    cv::merge(final_channels, 3, final);
    final.convertTo(final, CV_8UC3);

    cv::imwrite("../smoothed.jpg", final);

    return 0;
}