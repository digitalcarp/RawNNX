// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2025 Daniel Gao

#include "fmt/format.h"
#include "fmt/ranges.h"

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

void printHelp()
{
    fmt::println(R"(Usage: modnet <model_path> <image> [<output>]

    Model Code: https://github.com/ZHKKKe/MODNet
    )");
}

constexpr int WINDOW_WIDTH = 800;
constexpr int WINDOW_HEIGHT = 600;

struct Args {
    std::string_view model_path;
    int64_t tensor_size;
};

template <size_t SIZE>
using Shape = std::array<int64_t, SIZE>;

constexpr int TENSOR_MULTIPLE = 32;
constexpr int TENSOR_SIZE = 512;
constexpr int BATCH_SIZE = 1;
constexpr int RGB_NUM_CHANNELS = 3;
constexpr int MASK_NUM_CHANNELS = 1;

bool checkCudaSupport()
{
    std::vector<std::string> available = Ort::GetAvailableProviders();
    for (const auto& p : available) {
        if (p.find("CUDA") == 0) return true;
    }

    return false;
}

void appendCudaProvider(Ort::SessionOptions& session_options)
{
    const OrtApi& api = Ort::GetApi();
    OrtCUDAProviderOptionsV2* cuda_options = nullptr;
    Ort::ThrowOnError(api.CreateCUDAProviderOptions(&cuda_options));

    std::vector<const char*> keys{
        "device_id",
        // "gpu_mem_limit",
        "arena_extend_strategy", "cudnn_conv_algo_search", "do_copy_in_default_stream",
        "cudnn_conv_use_max_workspace"
        // "cudnn_conv1d_pad_to_nc1d"
    };
    std::vector<const char*> values{
        "0",
        // "2147483648",
        "kNextPowerOfTwo", "EXHAUSTIVE", "1", "1"
        // "1"
    };

    Ort::ThrowOnError(api.UpdateCUDAProviderOptions(cuda_options, keys.data(),
                                                    values.data(), keys.size()));
    Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_CUDA_V2(session_options,
                                                                        cuda_options));

    api.ReleaseCUDAProviderOptions(cuda_options);
}

auto inputShape(int64_t width, int64_t height)
{
    return Shape<4>{BATCH_SIZE, RGB_NUM_CHANNELS, height, width};
}

auto outputShape(int64_t width, int64_t height)
{
    return Shape<4>{BATCH_SIZE, MASK_NUM_CHANNELS, height, width};
}

struct Resize {
    double x_scale = 1.0;
    double y_scale = 1.0;

    uint32_t img_w = 0;
    uint32_t img_h = 0;
    uint32_t tensor_w = 0;
    uint32_t tensor_h = 0;
};

double divd(double x, double y) { return x / y; }

Resize computeResize(int img_w, int img_h)
{
    int tensor_w = img_w;
    int tensor_h = img_h;

    if (std::max(img_w, img_h) < TENSOR_SIZE || std::min(img_w, img_h) > TENSOR_SIZE) {
        if (img_w >= img_h) {
            tensor_h = TENSOR_SIZE;
            tensor_w = divd(img_w, img_h * TENSOR_SIZE);
        } else {
            tensor_w = TENSOR_SIZE;
            tensor_h = divd(img_h, img_w * TENSOR_SIZE);
        }
    }

    tensor_w -= tensor_w % 32;
    tensor_h -= tensor_h % 32;

    Resize resize;
    resize.img_w = img_w;
    resize.img_h = img_h;

    resize.tensor_w = tensor_w;
    resize.tensor_h = tensor_h;

    resize.x_scale = divd(tensor_w, img_w);
    resize.y_scale = divd(tensor_h, img_h);

    return resize;
}

std::vector<float> generateInputTensorData(const cv::Mat& img, const Resize& resize)
{
    cv::Size new_size{static_cast<int>(resize.tensor_w),
                      static_cast<int>(resize.tensor_h)};
    cv::Mat resized_img;
    cv::resize(img, resized_img, new_size, 0, 0, cv::INTER_CUBIC);

    const size_t area = resize.tensor_w * resize.tensor_h;
    std::vector<float> input(area * RGB_NUM_CHANNELS);

    const uchar* mat_data = static_cast<uchar*>(resized_img.data);
    const int channels = resized_img.channels();

    // constexpr uint32_t R_TRANSPOSE = 1;
    // constexpr uint32_t G_TRANSPOSE = 2;
    // constexpr uint32_t B_TRANSPOSE = 0;
    constexpr uint32_t R_TRANSPOSE = 0;
    constexpr uint32_t G_TRANSPOSE = 1;
    constexpr uint32_t B_TRANSPOSE = 2;

    // Normalize values to [-1, 1]
    auto convert = [](int value) -> float
    {
        // mean = 0.5, std = 0.5
        return std::clamp((value - 127.5f) / 127.5f, -1.f, 1.f);
    };

    for (uint32_t row = 0; row < resize.tensor_h; row++) {
        for (uint32_t col = 0; col < resize.tensor_w; col++) {
            // OpenCV image channel order is 8-bit BGR
            uchar b = mat_data[row * resized_img.step + channels * col + 0];
            uchar g = mat_data[row * resized_img.step + channels * col + 1];
            uchar r = mat_data[row * resized_img.step + channels * col + 2];

            auto offset = [&](uint32_t transpose)
            { return transpose * area + row * resize.tensor_w + col; };

            input[offset(R_TRANSPOSE)] = convert(r);
            input[offset(G_TRANSPOSE)] = convert(g);
            input[offset(B_TRANSPOSE)] = convert(b);
        }
    }

    return input;
}

cv::Mat readOutputTensorData(const std::vector<float>& output, const Resize& resize)
{
    cv::Mat mask(resize.tensor_h, resize.tensor_w, CV_8UC1);
    uchar* data = static_cast<uchar*>(mask.data);

    for (uint32_t row = 0; row < resize.tensor_h; row++) {
        for (uint32_t col = 0; col < resize.tensor_w; col++) {
            float value = output[row * resize.tensor_w + col];
            data[row * mask.step + col] = value * 255;
        }
    }

    cv::Mat result;
    cv::Size new_size{static_cast<int>(resize.img_w), static_cast<int>(resize.img_h)};
    cv::resize(mask, result, new_size, 0, 0, cv::INTER_LINEAR);

    return result;
}

class Demo {
public:
    void init(std::string_view model_path);
    cv::Mat run(const cv::Mat& img);

private:
    Ort::Env m_env{nullptr};
    Ort::Session m_session{nullptr};
    Ort::MemoryInfo m_mem_info{nullptr};
};

} // namespace

void Demo::init(std::string_view model_path)
{
    m_env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "demo");

    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    const bool use_cuda = checkCudaSupport();
    if (use_cuda) {
        appendCudaProvider(session_options);
    }

    m_session = Ort::Session(m_env, model_path.data(), session_options);

    if (use_cuda) {
        constexpr int device_id = 0; // Might not work if you have multiple GPUs
        m_mem_info =
                Ort::MemoryInfo("Cuda", OrtArenaAllocator, device_id, OrtMemTypeDefault);
    } else {
        m_mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    }
}

cv::Mat Demo::run(const cv::Mat& img)
{
    Resize resize = computeResize(img.size().width, img.size().height);

    fmt::println("Resizing input image from {}x{} to {}x{} (scale = [{:.5}, {:.5}])",
                 resize.img_w, resize.img_h, resize.tensor_w, resize.tensor_h,
                 resize.x_scale, resize.y_scale);

    // Vectors initialize to zero which is unnecessary for our use cases.
    // Consider using a structure with different construct semantics.
    std::vector<float> input = generateInputTensorData(img, resize);

    auto input_shape = inputShape(resize.tensor_w, resize.tensor_h);
    Ort::Value input_tensor =
            Ort::Value::CreateTensor(m_mem_info, input.data(), input.size(),
                                     input_shape.data(), input_shape.size());

    std::vector<float> output(resize.tensor_w * resize.tensor_h);
    auto output_shape = outputShape(resize.tensor_w, resize.tensor_h);
    Ort::Value output_tensor =
            Ort::Value::CreateTensor(m_mem_info, output.data(), output.size(),
                                     output_shape.data(), output_shape.size());

    constexpr int64_t NUM_INPUTS = 1;
    constexpr std::array<const char*, NUM_INPUTS> INPUT_NAMES{"input"};
    constexpr int64_t NUM_OUTPUTS = 1;
    constexpr std::array<const char*, NUM_OUTPUTS> OUTPUT_NAMES{"output"};

    Ort::RunOptions run_options;
    m_session.Run(run_options, INPUT_NAMES.data(), &input_tensor, NUM_INPUTS,
                  OUTPUT_NAMES.data(), &output_tensor, NUM_OUTPUTS);

    return readOutputTensorData(output, resize);
}

int main(int argc, char* argv[])
{
    if (argc != 3 && argc != 4) {
        printHelp();
        return -1;
    }

    if (std::string_view(argv[1]) == "--help") {
        printHelp();
        return 0;
    }

    const std::string_view model_path = argv[1];
    const std::string image_path = argv[2];

    const std::string out_img_path = (argc == 4) ? argv[3] : "";
    if (!out_img_path.empty()) {
        fmt::println("Mask will be written to: {}", out_img_path);
    }

    const cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR_BGR);
    cv::Mat mask;

    // Have the Ort::Session lifetime outlive the first waitKey(0) so it is
    // easier to check memory usage of the model.
    Demo demo;

    try {
        fmt::println("Setting up model...");
        demo.init(model_path);
        fmt::println("Done!");
        fmt::println("Running inference...");
        mask = demo.run(img);
        fmt::println("Done!");
    } catch (const std::exception& e) {
        fmt::println("{}", e.what());
        return -1;
    }

    if (!out_img_path.empty()) {
        fmt::println("Writing mask: {}", out_img_path);
        cv::imwrite(out_img_path, mask);
    }

    cv::Mat colored_mask(img.size(), img.type());
    for (int row = 0; row < img.rows; row++) {
        uchar* out = colored_mask.ptr<uchar>(row);
        uchar* in = mask.ptr<uchar>(row);
        for (int col = 0; col < img.cols; col++) {
            uchar mask_value = in[col];

            uint32_t offset = RGB_NUM_CHANNELS * static_cast<uint32_t>(col);
            // BGR format
            out[offset + 0] = 0;
            out[offset + 1] = mask_value;
            out[offset + 2] = mask_value;
        }
    }

    constexpr double alpha = 0.5;
    constexpr double beta = 1.0 - alpha;
    cv::Mat display;
    cv::addWeighted(img, alpha, colored_mask, beta, /*gamma*/ 0, display);

    const std::string INPUT_NAME = "MODNet Demo";
    cv::namedWindow(INPUT_NAME, cv::WINDOW_NORMAL);
    cv::resizeWindow(INPUT_NAME, WINDOW_WIDTH, WINDOW_HEIGHT);

    cv::imshow(INPUT_NAME, display);

    cv::waitKey(0);

    cv::destroyAllWindows();
    return 0;
}
