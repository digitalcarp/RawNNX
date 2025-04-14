// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2025 Daniel Gao

#include "fmt/format.h"
#include "fmt/ranges.h"

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

void printHelp()
{
    fmt::println(R"(Usage: mobilesam <encoder_path> <decoder_path> <image> [<output>]

    Model Code: https://github.com/ChaoningZhang/MobileSAM
    ONNX Model: https://huggingface.co/Acly/MobileSAM

    An image viewer will appear with the input image. You can select the inferencing prompts in this window.

    Click with the left mouse button to add a point prompt:
        Press the f key to switch to include point mode.
        Press the d key to switch to exclude point mode.

        The initial mode is the include mode.

    Press the r key to add a bounding box prompt.
        Click and drag the selection in the selection window.
        With the mouse key still held down, press space to confirm the selection.

        Only one bounding box prompt is allowed. Adding another will override any previous bounding box.

    Press the x key to run the inferencing model.
        After the segmentation completes, you can reprompt the model.
        A low resolution mask saved from the previous inference helps guide the next inference.

    Press the s key to save the full resolution mask to the given output file.

    Press the c key to reset the demo.
        Resetting the demo clears the low resolution mask from the last inference run.

    Press any other button to quit.
    )");
}

template <size_t SIZE>
using Shape = std::array<int64_t, SIZE>;

constexpr uint32_t RGB_NUM_CHANNELS = 3;
constexpr uint32_t ENCODER_SIZE = 1024;
constexpr int64_t LOW_RES_MASK_SIZE = 256;

constexpr Shape<4> ENCODED_SHAPE = {1, 256, 64, 64};
constexpr Shape<4> MASK_INPUT_SHAPE = {1, 1, LOW_RES_MASK_SIZE, LOW_RES_MASK_SIZE};
constexpr Shape<1> HAS_MASK_INPUT_SHAPE = {1};
constexpr Shape<1> ORIG_IMG_SIZE_SHAPE = {2};

using LowResMask = std::array<float, LOW_RES_MASK_SIZE * LOW_RES_MASK_SIZE>;

enum class Label : int {
    PADDING = -1,
    EXCLUDE = 0,
    INCLUDE = 1,
    BBOX_TOP_LEFT = 2,
    BBOX_BOT_RIGHT = 3
};

auto inputImageShape(uint32_t width, uint32_t height)
{
    return Shape<3>{height, width, 3};
}

std::pair<Shape<3>, Shape<2>> pointInputShapes(int64_t num_points)
{
    std::pair<Shape<3>, Shape<2>> result;
    result.first = {1, num_points, 2};
    result.second = {1, num_points};
    return result;
}

auto outputMaskShape(int num_masks, uint32_t width, uint32_t height)
{
    return Shape<4>{/*batch*/ 1, num_masks, height, width};
}

constexpr auto iouPredictionsShape(int num_masks) { return Shape<2>{num_masks, 1}; }

constexpr auto lowResMaskShape(int num_masks)
{
    return Shape<4>{num_masks, 1, LOW_RES_MASK_SIZE, LOW_RES_MASK_SIZE};
}

constexpr bool ENABLE_SHOW_LOW_RES_MASK = false;
constexpr int WINDOW_WIDTH = 800;
constexpr int WINDOW_HEIGHT = 600;

// Origin at top-left
struct Point {
    Point() : x(0), y(0) {}
    Point(float _x, float _y) : x(_x), y(_y) {}

    float x;
    float y;
};

struct BBox {
    BBox() = default;
    BBox(float x0, float y0, float x1, float y1)
            : top_left(std::min(x0, x1), std::min(y0, y1))
            , bot_right(std::max(x0, x1), std::max(y0, y1))
    {}

    Point top_left;
    Point bot_right;
};

struct Prompts {
    std::vector<Point> include_points;
    std::vector<Point> exclude_points;
    std::optional<BBox> bbox;
};

class Demo {
public:
    Demo(std::string_view image_path, std::string_view encoder_path,
         std::string_view decoder_path, std::string_view out_img_path)
            : m_orig_img(cv::imread(std::string(image_path), cv::IMREAD_COLOR_BGR))
            , m_encoder_path(encoder_path)
            , m_decoder_path(decoder_path)
            , m_out_img_path(out_img_path)
    {
        m_low_res_mask.fill(0);

        setupInference();

        m_orig_img.copyTo(m_display_img);
        // Output is an alpha mask
        m_output_mask.create(m_orig_img.size(), CV_8UC1);
    }

    const std::string& window() const { return m_window; }
    const cv::Mat& orig() const { return m_orig_img; }

    void runInference();

    bool handleKeyPress(int key);
    void addPoint(float x, float y);
    void selectBBox();

    void redraw() { cv::imshow(m_window, m_display_img); }
    void reset();

private:
    void setupInference();
    void encodeImage();
    std::pair<std::vector<float>, std::vector<float>> convertPrompts() const;

    void drawIncludePrompt(float x, float y);
    void drawExcludePrompt(float x, float y);
    void drawIncludePrompt(const Point& p) { drawIncludePrompt(p.x, p.y); }
    void drawExcludePrompt(const Point& p) { drawExcludePrompt(p.x, p.y); }
    void drawBBox(const BBox& bbox);
    void showOutput();
    void showLowResMask();

    const std::string m_window = "MobileSAM Demo";

    const cv::Mat m_orig_img;
    const std::string m_encoder_path;
    const std::string m_decoder_path;
    const std::string m_out_img_path;

    struct ResizeLongestSide {
        double scale = 1.0;
        uint32_t encoder_size = 1024;
        uint32_t old_width = 0;
        uint32_t old_height = 0;
        uint32_t new_width = 0;
        uint32_t new_height = 0;
    };
    ResizeLongestSide m_resize;

    LowResMask m_low_res_mask;
    cv::Mat m_output_mask;
    cv::Mat m_display_img;

    Prompts m_prompts;

    // ONNX Runtime
    struct EncodedImageTensor {
        Ort::MemoryInfo mem_info{nullptr};
        Ort::Allocator allocator{nullptr};
        Ort::Value tensor{nullptr};
    };
    Ort::Env m_env{nullptr};
    Ort::Session m_encoder_session{nullptr};
    Ort::Session m_decoder_session{nullptr};
    EncodedImageTensor m_encoded_image;

    bool m_is_include_mode = true;
    bool m_needs_infer = true;
    bool m_has_mask_input = false;
};

void handleMouseEvent(int action, int x, int y, int flags, void* userdata)
{
    auto demo = static_cast<Demo*>(userdata);

    if (action == cv::EVENT_LBUTTONDOWN) {
        demo->addPoint(x, y);
    }
}

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

} // namespace

void Demo::addPoint(float x, float y)
{
    if (!m_needs_infer) {
        m_prompts.include_points.clear();
        m_prompts.exclude_points.clear();
        m_prompts.bbox = std::nullopt;
        showOutput();
    }

    if (m_is_include_mode) {
        m_prompts.include_points.emplace_back(x, y);
        m_needs_infer = true;
        drawIncludePrompt(x, y);
        redraw();
    } else {
        m_prompts.exclude_points.emplace_back(x, y);
        m_needs_infer = true;
        drawExcludePrompt(x, y);
        redraw();
    }
}

void Demo::selectBBox()
{
    const std::string name = "Select Bounding Box";
    cv::namedWindow(name, cv::WINDOW_NORMAL);
    cv::resizeWindow(name, WINDOW_WIDTH, WINDOW_HEIGHT);

    cv::Rect rect = cv::selectROI(name, m_display_img, /*crosshair*/ false);
    cv::destroyWindow(name);

    if (!rect.empty()) {
        if (!m_needs_infer) {
            m_prompts.include_points.clear();
            m_prompts.exclude_points.clear();
            m_prompts.bbox = std::nullopt;
            showOutput();
        }

        BBox bbox(rect.tl().x, rect.tl().y, rect.br().x, rect.br().y);
        m_prompts.bbox = bbox;
        m_needs_infer = true;
        drawBBox(bbox);
        redraw();
    }
}

void Demo::drawIncludePrompt(float x, float y)
{
    cv::Point p(x, y);
    cv::circle(m_display_img, p, 20, {0, 255, 0}, cv::FILLED);
}

void Demo::drawExcludePrompt(float x, float y)
{
    cv::Point p(x, y);
    cv::circle(m_display_img, p, 20, {0, 0, 255}, cv::FILLED);
}

void Demo::drawBBox(const BBox& bbox)
{
    cv::Point tl(bbox.top_left.x, bbox.top_left.y);
    cv::Point br(bbox.bot_right.x, bbox.bot_right.y);
    cv::Rect rect(tl, br);
    cv::rectangle(m_display_img, rect, {255, 0, 0}, 20);
}

void Demo::showOutput()
{
    cv::Mat colored_mask = cv::Mat::zeros(m_orig_img.size(), m_orig_img.type());
    colored_mask.setTo(cv::Scalar(0, 255, 255), m_output_mask);

    constexpr double alpha = 0.5;
    constexpr double beta = 1.0 - alpha;
    cv::addWeighted(m_orig_img, alpha, colored_mask, beta, /*gamma*/ 0, m_display_img);

    for (const auto& p : m_prompts.include_points) {
        drawIncludePrompt(p);
    }
    for (const auto& p : m_prompts.exclude_points) {
        drawExcludePrompt(p);
    }
    if (m_prompts.bbox) {
        drawBBox(*m_prompts.bbox);
    }

    redraw();
}

void Demo::showLowResMask()
{
    cv::Mat low_res = cv::Mat::zeros(LOW_RES_MASK_SIZE, LOW_RES_MASK_SIZE, CV_8UC1);
    uchar* data = static_cast<uchar*>(low_res.data);

    for (uint32_t row = 0; row < LOW_RES_MASK_SIZE; row++) {
        for (uint32_t col = 0; col < LOW_RES_MASK_SIZE; col++) {
            uint8_t value = m_low_res_mask[row * LOW_RES_MASK_SIZE + col];
            data[row * low_res.step + col] = value;
        }
    }

    const std::string name = "Low Resolution Mask";
    cv::namedWindow(name, cv::WINDOW_NORMAL);
    cv::resizeWindow(name, WINDOW_WIDTH, WINDOW_HEIGHT);
    cv::imshow(name, low_res);
}

void Demo::reset()
{
    m_orig_img.copyTo(m_display_img);
    m_output_mask.setTo(cv::Scalar(0));

    m_prompts.include_points.clear();
    m_prompts.exclude_points.clear();
    m_prompts.bbox = std::nullopt;

    m_needs_infer = true;
    m_has_mask_input = false;
    m_low_res_mask.fill(0);

    redraw();
}

bool Demo::handleKeyPress(int key)
{
    switch (key) {
    case 'f':
        m_is_include_mode = true;
        return true;
    case 'd':
        m_is_include_mode = false;
        return true;
    case 'r':
        selectBBox();
        return true;
    case 'x':
        runInference();
        return true;
    case 'c':
        reset();
        return true;
    case 's':
        if (!m_out_img_path.empty()) {
            fmt::println("Writing mask to: {}", m_out_img_path);
            cv::imwrite(m_out_img_path, m_output_mask);
            return false;
        } else {
            fmt::println("Write failed: no output path provided");
            return true;
        }
    default:
        break;
    }

    return false;
}

void Demo::setupInference()
{
    m_env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "demo");

    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    const bool use_cuda = checkCudaSupport();
    if (use_cuda) {
        appendCudaProvider(session_options);
    }

    m_encoder_session = Ort::Session(m_env, m_encoder_path.c_str(), session_options);
    m_decoder_session = Ort::Session(m_env, m_decoder_path.c_str(), session_options);

    m_encoded_image = EncodedImageTensor{};

    if (use_cuda) {
        constexpr int device_id = 0; // Might not work if you have multiple GPUs
        m_encoded_image.mem_info =
                Ort::MemoryInfo("Cuda", OrtArenaAllocator, device_id, OrtMemTypeDefault);
    } else {
        m_encoded_image.mem_info =
                Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    }

    m_encoded_image.allocator =
            Ort::Allocator(m_encoder_session, m_encoded_image.mem_info);
    m_encoded_image.tensor = Ort::Value::CreateTensor(
            m_encoded_image.allocator, ENCODED_SHAPE.data(), ENCODED_SHAPE.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    encodeImage();
}

void Demo::encodeImage()
{
    auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Allocator allocator(m_encoder_session, mem_info);

    m_resize = ResizeLongestSide{};
    m_resize.encoder_size = ENCODER_SIZE;
    m_resize.old_width = m_orig_img.size().width;
    m_resize.old_height = m_orig_img.size().height;

    // One side of the image must be ENCODER_SIZE (1024px) long.
    m_resize.scale =
            m_resize.encoder_size
            / static_cast<double>(std::max(m_resize.old_width, m_resize.old_height));

    m_resize.new_width = m_resize.scale * m_resize.old_width;
    m_resize.new_height = m_resize.scale * m_resize.old_height;

    fmt::println("Resizing input image from {}x{} to {}x{} (scale = {:.5})",
                 m_resize.old_width, m_resize.old_height, m_resize.new_width,
                 m_resize.new_height, m_resize.scale);

    cv::Mat resized_img;
    {
        // Alternative for shrinking is Catmull-Rom bicubic interpolation.
        // It behaves better when shrinking a lot and sharpens the image. For a RAW
        // image of 6000x4000, it would need to be resized to within 1024x1024.
        auto interp = m_resize.scale >= 1.0 ? cv::INTER_LANCZOS4 : cv::INTER_AREA;

        cv::Size new_size(m_resize.new_width, m_resize.new_height);
        double scale = m_resize.scale;

        cv::resize(m_orig_img, resized_img, new_size, scale, scale, interp);
    }

    if (resized_img.cols != m_resize.new_width)
        throw std::runtime_error("Wrong width after resize");
    if (resized_img.rows != m_resize.new_height)
        throw std::runtime_error("Wrong height after resize");

    auto shape = inputImageShape(m_resize.new_width, m_resize.new_height);
    auto input_image_tensor =
            Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());
    float* data = input_image_tensor.GetTensorMutableData<float>();

    const uchar* mat_data = static_cast<uchar*>(resized_img.data);
    const int channels = resized_img.channels();
    auto convert = [](int value) -> float { return value; };

    for (uint32_t row = 0; row < m_resize.new_height; row++) {
        for (uint32_t col = 0; col < m_resize.new_width; col++) {
            // OpenCV image channel order is 8-bit BGR
            uchar b = mat_data[row * resized_img.step + channels * col + 0];
            uchar g = mat_data[row * resized_img.step + channels * col + 1];
            uchar r = mat_data[row * resized_img.step + channels * col + 2];

            uint32_t pixel = RGB_NUM_CHANNELS * (row * m_resize.new_width + col);
            data[pixel + 0] = convert(r);
            data[pixel + 1] = convert(g);
            data[pixel + 2] = convert(b);
        }
    }

    Ort::IoBinding io_binding(m_encoder_session);
    io_binding.BindInput("input_image", input_image_tensor);
    io_binding.BindOutput("image_embeddings", m_encoded_image.tensor);

    fmt::print("Running MobileSAM image encoder... ");

    Ort::RunOptions run_options;
    m_encoder_session.Run(run_options, io_binding);

    fmt::println("Done!");
}

void Demo::runInference()
{
    if (!m_needs_infer) return;

    fmt::print("Running MobileSAM decoder with prompts... ");

    auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto allocator = Ort::Allocator(m_decoder_session, mem_info);

    auto [coords, labels] = convertPrompts();
    auto [coords_shape, labels_shape] = pointInputShapes(labels.size());

    auto point_coords_tensor =
            Ort::Value::CreateTensor(mem_info, coords.data(), coords.size(),
                                     coords_shape.data(), coords_shape.size());

    auto point_labels_tensor =
            Ort::Value::CreateTensor(mem_info, labels.data(), labels.size(),
                                     labels_shape.data(), labels_shape.size());

    auto mask_input_tensor = Ort::Value::CreateTensor(
            mem_info, m_low_res_mask.data(), m_low_res_mask.size(),
            MASK_INPUT_SHAPE.data(), MASK_INPUT_SHAPE.size());

    std::array<float, 1> has_mask_input{m_has_mask_input ? 0.f : 1.f};
    auto has_mask_input_tensor = Ort::Value::CreateTensor(
            mem_info, has_mask_input.data(), has_mask_input.size(),
            HAS_MASK_INPUT_SHAPE.data(), HAS_MASK_INPUT_SHAPE.size());

    std::array<float, 2> orig_im_size{static_cast<float>(m_resize.old_height),
                                      static_cast<float>(m_resize.old_width)};
    auto orig_im_size_tensor = Ort::Value::CreateTensor(
            mem_info, orig_im_size.data(), orig_im_size.size(),
            ORIG_IMG_SIZE_SHAPE.data(), ORIG_IMG_SIZE_SHAPE.size());

    constexpr int NUM_MASKS = 1;

    auto masks_output_shape =
            outputMaskShape(NUM_MASKS, m_resize.old_width, m_resize.old_height);
    auto masks_output_tensor = Ort::Value::CreateTensor<float>(
            allocator, masks_output_shape.data(), masks_output_shape.size());

    auto iou_shape = iouPredictionsShape(NUM_MASKS);
    auto iou_tensor = Ort::Value::CreateTensor<float>(allocator, iou_shape.data(),
                                                      iou_shape.size());

    auto low_res_mask_shape = lowResMaskShape(NUM_MASKS);
    auto low_res_mask_tensor = Ort::Value::CreateTensor<float>(
            mem_info, m_low_res_mask.data(), m_low_res_mask.size(),
            low_res_mask_shape.data(), low_res_mask_shape.size());

    Ort::IoBinding io_binding(m_decoder_session);
    io_binding.BindInput("image_embeddings", m_encoded_image.tensor);
    io_binding.BindInput("point_coords", point_coords_tensor);
    io_binding.BindInput("point_labels", point_labels_tensor);
    io_binding.BindInput("mask_input", mask_input_tensor);
    io_binding.BindInput("has_mask_input", has_mask_input_tensor);
    io_binding.BindInput("orig_im_size", orig_im_size_tensor);
    io_binding.BindOutput("masks", masks_output_tensor);
    io_binding.BindOutput("iou_predictions", iou_tensor);
    io_binding.BindOutput("low_res_masks", low_res_mask_tensor);

    Ort::RunOptions run_options;
    m_decoder_session.Run(run_options, io_binding);

    const float* masks_output = masks_output_tensor.GetTensorData<float>();

    uint32_t width = m_resize.old_width;
    uint32_t height = m_resize.old_height;
    uchar* out_data = static_cast<uchar*>(m_output_mask.data);

    for (uint32_t row = 0; row < height; row++) {
        for (uint32_t col = 0; col < width; col++) {
            uint8_t value = masks_output[row * width + col] > 0.0 ? 0xFF : 0;
            out_data[row * m_output_mask.step + col] = value;
        }
    }

    fmt::println("Done! (iou = {:.5})", iou_tensor.GetTensorData<float>()[0]);

    m_needs_infer = false;
    m_has_mask_input = true;

    showOutput();
    if constexpr (ENABLE_SHOW_LOW_RES_MASK) {
        showLowResMask();
    }
}

std::pair<std::vector<float>, std::vector<float>> Demo::convertPrompts() const
{
    std::vector<float> coords;
    std::vector<float> labels;

    auto add = [&](const Point& p, Label label)
    {
        coords.push_back(p.x * m_resize.scale);
        coords.push_back(p.y * m_resize.scale);
        labels.push_back(static_cast<float>(label));
    };

    for (auto& p : m_prompts.include_points) {
        add(p, Label::INCLUDE);
    }

    for (auto& p : m_prompts.exclude_points) {
        add(p, Label::EXCLUDE);
    }

    if (m_prompts.bbox) {
        add(m_prompts.bbox->top_left, Label::BBOX_TOP_LEFT);
        add(m_prompts.bbox->bot_right, Label::BBOX_BOT_RIGHT);
    }

    if (labels.empty()) {
        // Add a bbox around the entire image if not prompts are given
        add({0, 0}, Label::BBOX_TOP_LEFT);
        auto new_w = static_cast<float>(m_resize.new_width);
        auto new_h = static_cast<float>(m_resize.new_height);
        add({new_w, new_h}, Label::BBOX_BOT_RIGHT);
    } else if (!m_prompts.bbox) {
        // Need to append a padding point if no bbox
        add({0, 0}, Label::PADDING);
    }

    return {std::move(coords), std::move(labels)};
}

int main(int argc, char* argv[])
{
    if (argc != 4 && argc != 5) {
        printHelp();
        return -1;
    }

    if (std::string_view(argv[1]) == "--help") {
        printHelp();
        return 0;
    }

    std::string_view encoder_path = argv[1];
    std::string_view decoder_path = argv[2];
    std::string_view img_path = argv[3];

    std::string_view out_img_path = "";
    if (argc == 5) {
        out_img_path = argv[4];
        fmt::println("Mask may be written to: {}", out_img_path);
    }

    Demo demo(img_path, encoder_path, decoder_path, out_img_path);

    cv::namedWindow(demo.window(), cv::WINDOW_NORMAL);
    cv::resizeWindow(demo.window(), WINDOW_WIDTH, WINDOW_HEIGHT);

    cv::setMouseCallback(demo.window(), handleMouseEvent, &demo);

    demo.redraw();

    bool keep_going = true;
    while (keep_going) {
        int key = cv::waitKey(0);
        keep_going = demo.handleKeyPress(key);
    }

    return 0;
}
