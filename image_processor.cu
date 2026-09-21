#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#define CUDA_CHECK(call)                                                   \
    do {                                                                    \
        cudaError_t error = call;                                           \
        if (error != cudaSuccess) {                                         \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error)        \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return 1;                                                       \
        }                                                                   \
    } while (0)

struct Image {
    int width;
    int height;
    std::vector<unsigned char> data;
};

/*
 * CUDA kernel:
 *
 * Each CUDA thread processes one pixel.
 *
 * The kernel performs a 3x3 Gaussian blur followed by grayscale conversion.
 */
__global__ void processImageKernel(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    /*
     * 3x3 Gaussian filter:
     *
     * 1 2 1
     * 2 4 2
     * 1 2 1
     *
     * divided by 16
     */

    const int kernel[3][3] = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    };

    float red = 0.0f;
    float green = 0.0f;
    float blue = 0.0f;
    int weightSum = 0;

    for (int ky = -1; ky <= 1; ky++) {

        for (int kx = -1; kx <= 1; kx++) {

            int px = min(max(x + kx, 0), width - 1);
            int py = min(max(y + ky, 0), height - 1);

            int index = (py * width + px) * 3;

            int weight = kernel[ky + 1][kx + 1];

            red += input[index] * weight;
            green += input[index + 1] * weight;
            blue += input[index + 2] * weight;

            weightSum += weight;
        }
    }

    red /= weightSum;
    green /= weightSum;
    blue /= weightSum;

    /*
     * Convert filtered pixel to grayscale.
     */
    unsigned char gray =
        static_cast<unsigned char>(
            0.299f * red +
            0.587f * green +
            0.114f * blue
        );

    int outputIndex = (y * width + x) * 3;

    output[outputIndex] = gray;
    output[outputIndex + 1] = gray;
    output[outputIndex + 2] = gray;
}


/*
 * Read a binary PPM (P6) image.
 */
bool readPPM(const std::string& filename, Image& image)
{
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "Unable to open " << filename << std::endl;
        return false;
    }

    std::string magic;

    file >> magic;

    if (magic != "P6") {
        std::cerr << "Unsupported image format: " << filename << std::endl;
        return false;
    }

    file >> image.width;
    file >> image.height;

    int maxValue;
    file >> maxValue;

    file.get();

    if (maxValue != 255) {
        std::cerr << "Unsupported PPM maximum value." << std::endl;
        return false;
    }

    size_t dataSize =
        static_cast<size_t>(image.width) *
        static_cast<size_t>(image.height) *
        3;

    image.data.resize(dataSize);

    file.read(
        reinterpret_cast<char*>(image.data.data()),
        dataSize
    );

    return file.good();
}


/*
 * Write binary PPM (P6).
 */
bool writePPM(
    const std::string& filename,
    const Image& image)
{
    std::ofstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "Unable to create " << filename << std::endl;
        return false;
    }

    file << "P6\n";
    file << image.width << " " << image.height << "\n";
    file << "255\n";

    file.write(
        reinterpret_cast<const char*>(image.data.data()),
        image.data.size()
    );

    return file.good();
}


int main()
{
    const std::string inputDirectory = "input";
    const std::string outputDirectory = "output";
    const std::string resultsDirectory = "results";

    fs::create_directories(inputDirectory);
    fs::create_directories(outputDirectory);
    fs::create_directories(resultsDirectory);

    std::vector<fs::path> imageFiles;

    for (const auto& entry : fs::directory_iterator(inputDirectory)) {

        if (!entry.is_regular_file())
            continue;

        if (entry.path().extension() == ".ppm") {
            imageFiles.push_back(entry.path());
        }
    }

    std::sort(imageFiles.begin(), imageFiles.end());

    if (imageFiles.empty()) {

        std::cerr << "No PPM images found in input/" << std::endl;
        std::cerr << "Run: python3 generate_images.py" << std::endl;

        return 1;
    }

    std::cout << "========================================\n";
    std::cout << " CUDA Image Processing at Scale\n";
    std::cout << "========================================\n\n";

    std::cout << "Images found: " << imageFiles.size() << "\n";

    /*
     * Check CUDA device.
     */
    int deviceCount = 0;

    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "No CUDA GPU detected." << std::endl;
        return 1;
    }

    cudaDeviceProp prop;

    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Compute capability: "
              << prop.major << "." << prop.minor << "\n";

    std::cout << "Global memory: "
              << prop.totalGlobalMem / (1024 * 1024)
              << " MB\n\n";

    /*
     * CUDA events provide GPU-side timing.
     */
    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    double totalGpuTime = 0.0;
    int successfulImages = 0;

    for (size_t imageNumber = 0;
         imageNumber < imageFiles.size();
         imageNumber++)
    {
        const std::string inputFilename =
            imageFiles[imageNumber].string();

        Image inputImage;

        if (!readPPM(inputFilename, inputImage)) {
            std::cerr << "Skipping image: "
                      << inputFilename << std::endl;
            continue;
        }

        Image outputImage;

        outputImage.width = inputImage.width;
        outputImage.height = inputImage.height;

        outputImage.data.resize(
            inputImage.data.size()
        );

        size_t bytes =
            inputImage.data.size() *
            sizeof(unsigned char);

        unsigned char* d_input = nullptr;
        unsigned char* d_output = nullptr;

        CUDA_CHECK(
            cudaMalloc(
                &d_input,
                bytes
            )
        );

        CUDA_CHECK(
            cudaMalloc(
                &d_output,
                bytes
            )
        );

        CUDA_CHECK(
            cudaMemcpy(
                d_input,
                inputImage.data.data(),
                bytes,
                cudaMemcpyHostToDevice
            )
        );

        dim3 blockSize(16, 16);

        dim3 gridSize(
            (inputImage.width + blockSize.x - 1) /
                blockSize.x,

            (inputImage.height + blockSize.y - 1) /
                blockSize.y
        );

        CUDA_CHECK(cudaEventRecord(start));

        processImageKernel<<<gridSize, blockSize>>>(
            d_input,
            d_output,
            inputImage.width,
            inputImage.height
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));

        CUDA_CHECK(cudaEventSynchronize(stop));

        float milliseconds = 0.0f;

        CUDA_CHECK(
            cudaEventElapsedTime(
                &milliseconds,
                start,
                stop
            )
        );

        totalGpuTime += milliseconds;

        CUDA_CHECK(
            cudaMemcpy(
                outputImage.data.data(),
                d_output,
                bytes,
                cudaMemcpyDeviceToHost
            )
        );

        fs::path outputPath =
            fs::path(outputDirectory) /
            imageFiles[imageNumber].filename();

        if (!writePPM(outputPath.string(), outputImage)) {

            cudaFree(d_input);
            cudaFree(d_output);

            continue;
        }

        cudaFree(d_input);
        cudaFree(d_output);

        successfulImages++;

        std::cout
            << "Processed "
            << std::setw(4)
            << successfulImages
            << "/"
            << imageFiles.size()
            << " : "
            << imageFiles[imageNumber].filename()
            << " | GPU kernel: "
            << std::fixed
            << std::setprecision(3)
            << milliseconds
            << " ms\n";
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    double averageTime =
        successfulImages > 0
        ? totalGpuTime / successfulImages
        : 0.0;

    std::ofstream logFile(
        resultsDirectory + "/execution_log.txt"
    );

    logFile << "CUDA Image Processing at Scale\n";
    logFile << "====================================\n\n";

    logFile << "GPU: "
            << prop.name
            << "\n";

    logFile << "Compute Capability: "
            << prop.major
            << "."
            << prop.minor
            << "\n";

    logFile << "Images discovered: "
            << imageFiles.size()
            << "\n";

    logFile << "Images successfully processed: "
            << successfulImages
            << "\n";

    logFile << "Total GPU kernel time: "
            << std::fixed
            << std::setprecision(3)
            << totalGpuTime
            << " ms\n";

    logFile << "Average GPU kernel time/image: "
            << averageTime
            << " ms\n";

    logFile.close();

    std::cout << "\n========================================\n";
    std::cout << "Processing complete\n";
    std::cout << "========================================\n";

    std::cout
        << "Successfully processed: "
        << successfulImages
        << "/"
        << imageFiles.size()
        << "\n";

    std::cout
        << "Total GPU kernel time: "
        << std::fixed
        << std::setprecision(3)
        << totalGpuTime
        << " ms\n";

    std::cout
        << "Average GPU time/image: "
        << averageTime
        << " ms\n";

    std::cout
        << "Output directory: output/\n";

    std::cout
        << "Execution log: results/execution_log.txt\n";

    return 0;
}
