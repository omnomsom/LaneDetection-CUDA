#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <math.h>
#include <opencv2/opencv.hpp>

#define DIVISIONS 360
#define THRESHOLD 400
#define NUM_LINES_TO_VISUALIZE 20

using namespace cv;
using namespace std;

__device__ int8_t canny_x[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
__device__ int8_t canny_y[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

__global__ void canny_edge_detection(const uint8_t* input, uint8_t* output, int width, int height, int low_threshold, int high_threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = (y * width + x) * 3;

    if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
        int gx = 0;
        int gy = 0;
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                int pixel_value = input[((y + i) * width + x + j) * 3];
                gx += pixel_value * canny_x[i + 1][j + 1];
                gy += pixel_value * canny_y[i + 1][j + 1];
            }
        }

        int magnitude = sqrtf(gx * gx + gy * gy);
        uint8_t edge_value = (magnitude > high_threshold) ? 255 : ((magnitude < low_threshold) ? 0 : magnitude);
        output[idx] = edge_value;
        output[idx + 1] = edge_value;
        output[idx + 2] = edge_value;
    }
}

int main() {
    // Load image using OpenCV
    Mat image = imread("lanes.jpg", IMREAD_COLOR);

    // Check if the image is loaded successfully
    if (image.empty()) {
        fprintf(stderr, "Error: Couldn't load the image.\n");
        return -1;
    }

    // Apply Gaussian blur
    GaussianBlur(image, image, cv::Size(11, 11), 2);  // Adjust kernel size and standard deviation as needed


    // Get image information
    int width = image.cols;
    int height = image.rows;
    int size = width * height * 3;  // Assuming 3 channels (RGB)

    // Allocate device memory
    uint8_t* d_input, * d_output;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    // Copy input image to device
    cudaMemcpy(d_input, image.data, size, cudaMemcpyHostToDevice);

    // Set up grid and block sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch Canny edge detection kernel
    canny_edge_detection << <gridSize, blockSize >> > (d_input, d_output, width, height, 30, 100);  // Adjust low and high thresholds as needed

    // Copy output back to host
    uint8_t* output = (uint8_t*)malloc(size);
    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);


    // Perform Hough Transformation on the result of Canny edge detection
    Mat cannyResult(height, width, CV_8UC3, output);
    Mat grayCannyResult;
    cvtColor(cannyResult, grayCannyResult, COLOR_BGR2GRAY);
    vector<Vec2f> lines;
    imwrite("cannyres.jpg", grayCannyResult);
    HoughLines(grayCannyResult, lines, 1, CV_PI / 180, 100);  // Adjust parameters as needed

    // Visualize only a subset of lines (e.g., 10 lines)
    Mat frame = image.clone();  // Clone the original image for visualization
    int numLinesToVisualize = min(NUM_LINES_TO_VISUALIZE, static_cast<int>(lines.size()));
    for (int i = 0; i < numLinesToVisualize; ++i) {
        float rho = lines[i][0], theta = lines[i][1];
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;

        Point pt1, pt2;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));

        line(frame, pt1, pt2, Scalar(255, 0, 0), 1, LINE_AA);
    }

    // Save the result using OpenCV
    imwrite("res.jpg", frame);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    free(output);

    return 0;
}
