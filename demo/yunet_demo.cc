#include "fmt/format.h"
#include "fmt/ranges.h"

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
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
    fmt::println(R"(Usage: yunet <model_path> <tensor_size> <conf> <nms> <image> [<output>]

    tensor_size    The width and height dimensions of the model.
                   Use an integer <= 0 for automatic sizing based on image size.
    conf           Confidence threshold for filtering out < conf
    nms            NMS threshold for reducing overlap

    Model: https://github.com/ShiqiYu/libfacedetection.train

    Example: yunet model.onnx 640 0.9 0.3 image.jpg
    )");
}

constexpr int WINDOW_WIDTH = 800;
constexpr int WINDOW_HEIGHT = 600;

template <size_t SIZE>
using Shape = std::array<int64_t, SIZE>;

constexpr int BATCH_SIZE = 1;
constexpr int RGB_NUM_CHANNELS = 3;
// constexpr int TENSOR_SIZE = 640;
constexpr int TENSOR_MULTIPLE = 32;

constexpr Shape<4> inputShape(int tensor_w, int tensor_h)
{
    return Shape<4>{BATCH_SIZE, RGB_NUM_CHANNELS, tensor_h, tensor_w};
}
constexpr Shape<3> clsShape(int feature_size)
{
    return Shape<3>{BATCH_SIZE, feature_size, 1};
}
constexpr Shape<3> objShape(int feature_size)
{
    return Shape<3>{BATCH_SIZE, feature_size, 1};
}
constexpr Shape<3> bboxShape(int feature_size)
{
    return Shape<3>{BATCH_SIZE, feature_size, 4};
}
constexpr Shape<3> kpsShape(int feature_size)
{
    return Shape<3>{BATCH_SIZE, feature_size, 10};
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
        "arena_extend_strategy",
        "cudnn_conv_algo_search",
        "do_copy_in_default_stream",
        "cudnn_conv_use_max_workspace"
        // "cudnn_conv1d_pad_to_nc1d"
    };
    std::vector<const char*> values{
        "0",
        // "2147483648",
        "kNextPowerOfTwo",
        "EXHAUSTIVE",
        "1",
        "1"
        // "1"
    };

    Ort::ThrowOnError(api.UpdateCUDAProviderOptions(
        cuda_options, keys.data(), values.data(), keys.size()));
    Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_CUDA_V2(
        session_options, cuda_options));

    api.ReleaseCUDAProviderOptions(cuda_options);
}

struct Resize {
    double x_scale = 1.0;
    double y_scale = 1.0;

    uint32_t img_w = 0;
    uint32_t img_h = 0;
    uint32_t new_w = 0;
    uint32_t new_h = 0;
    uint32_t tensor_w = 0;
    uint32_t tensor_h = 0;
};

double divd(double x, double y) { return x / y; }

int constrainToMultipleMin(double x, int multiple, int min)
{
    int y = static_cast<int>(std::round(x / multiple) * multiple);
    if (y < min) {
        y = static_cast<int>(std::ceil(x / multiple) * multiple);
    }
    return y;
}

Resize computeResize(int img_w, int img_h, int tensor_size)
{
    Resize resize;
    resize.img_w = img_w;
    resize.img_h = img_h;

    if (tensor_size > 0) {
        double scale_w = divd(tensor_size, img_w);
        double scale_h = divd(tensor_size, img_h);

        // Scale with tensor_size as upper bound
        if (scale_w < scale_h) {
            // Fit width
            scale_h = scale_w;
        } else {
            // Fit height
            scale_w = scale_h;
        }

        resize.new_w = img_w * scale_w;
        resize.new_h = img_h * scale_h;
        resize.x_scale = divd(resize.new_w, resize.img_w);
        resize.y_scale = divd(resize.new_h, resize.img_h);
        resize.tensor_w = tensor_size;
        resize.tensor_h = tensor_size;
    } else {
        resize.new_w = img_w;
        resize.new_h = img_h;
        resize.x_scale = 1.0;
        resize.y_scale = 1.0;
        resize.tensor_w = constrainToMultipleMin(img_w, TENSOR_MULTIPLE, img_w);
        resize.tensor_h = constrainToMultipleMin(img_h, TENSOR_MULTIPLE, img_h);
    }

    return resize;
}

std::vector<float> generateInputTensorData(const cv::Mat& img, const Resize& resize)
{
    cv::Mat resized_img;
    if ((resize.new_w == resize.img_w) && (resize.new_h == resize.img_h)) {
        resized_img = img;
    } else {
        cv::Size new_size{static_cast<int>(resize.new_w), static_cast<int>(resize.new_h)};
        cv::resize(img, resized_img, new_size, 0, 0, cv::INTER_AREA);
    }

    const size_t area = resize.tensor_w * resize.tensor_h;
    // Zero initialize data needed in case image needs padding with 0
    std::vector<float> input(area * RGB_NUM_CHANNELS, 0);

    const uchar* mat_data = static_cast<uchar*>(resized_img.data);
    const int channels = resized_img.channels();

    constexpr uint32_t R_TRANSPOSE = 2;
    constexpr uint32_t G_TRANSPOSE = 1;
    constexpr uint32_t B_TRANSPOSE = 0;

    auto convert = [&](int value) -> float { return value; };

    for (uint32_t row = 0; row < resize.new_h; row++) {
        for (uint32_t col = 0; col < resize.new_w; col++) {
            // OpenCV image channel order is 8-bit BGR
            uchar b = mat_data[row * resized_img.step + channels * col + 0];
            uchar g = mat_data[row * resized_img.step + channels * col + 1];
            uchar r = mat_data[row * resized_img.step + channels * col + 2];

            auto offset = [&](uint32_t transpose) {
                return transpose * area + row * resize.tensor_w + col;
            };

            input[offset(R_TRANSPOSE)] = convert(r);
            input[offset(G_TRANSPOSE)] = convert(g);
            input[offset(B_TRANSPOSE)] = convert(b);
        }
    }

    return input;
}

struct FeatureMap {
    // sqrt(cls * obj) gives the confidence
    std::vector<float> cls;
    std::vector<float> obj;
    std::vector<float> bbox;
    // 5 landmarks
    std::vector<float> kps;
    size_t cls_index;
    size_t obj_index;
    size_t bbox_index;
    size_t kps_index;
    size_t num_features;
    size_t width;
    size_t height;
    size_t stride;
};

FeatureMap generateFeatureMap(size_t stride, std::vector<Ort::Value>& tensors,
                              const Ort::MemoryInfo& mem_info, const Resize& resize)
{
    size_t feature_width = resize.tensor_w / stride;
    size_t feature_height = resize.tensor_h / stride;

    size_t num_features = feature_width * feature_height;

    auto cls_shape = clsShape(num_features);
    auto obj_shape = objShape(num_features);
    auto bbox_shape = bboxShape(num_features);
    auto kps_shape = kpsShape(num_features);

    // Sets the data backing array, appends the tensor, and returns the index
    auto create = [&](std::vector<float>& data, const auto& shape) -> size_t {
        size_t total_len = 1;
        for (auto dim : shape) {
            total_len *= dim;
        }
        data.resize(total_len, 0.f);

        size_t index = tensors.size();
        tensors.push_back(Ort::Value::CreateTensor(
            mem_info, data.data(), data.size(), shape.data(), shape.size()));

        return index;
    };

    FeatureMap map;
    map.cls_index = create(map.cls, cls_shape);
    map.obj_index = create(map.obj, obj_shape);
    map.bbox_index = create(map.bbox, bbox_shape);
    map.kps_index = create(map.kps, kps_shape);
    map.num_features = num_features;
    map.width = feature_width;
    map.height = feature_height;
    map.stride = stride;

    return map;
}

struct Point {
    float x;
    float y;
};

struct BBox {
    float x_min;
    float y_min;
    float x_max;
    float y_max;
};

struct Landmarks {
    Point right_eye;
    Point left_eye;
    Point nose_tip;
    Point right_mouth;
    Point left_mouth;
};

struct Faces {
    std::vector<BBox> bboxes;
    std::vector<Landmarks> landmarks;
    std::vector<float> scores;
};

void parseBBoxScore(const FeatureMap& feat, float conf_threshold, Faces& faces)
{
    const float fstride = feat.stride;

    constexpr size_t BBOX_DIM = 4;
    constexpr size_t KPS_DIM = 10;

    for (size_t r = 0; r < feat.height; r++) {
        const float prior_box_y = r * fstride;

        for (size_t c = 0; c < feat.width; c++) {
            const float prior_box_x = c * fstride;
            const size_t offset = r * feat.width + c;

            float cls_score = std::clamp(feat.cls[offset], 0.f, 1.f);
            float obj_score = std::clamp(feat.obj[offset], 0.f, 1.f);
            float score = std::sqrt(cls_score * obj_score);

            if (score < conf_threshold) continue;

            // Decode bbox using prior box
            const size_t bbox_offset = offset * BBOX_DIM;
            // clang-format off
            float cx =          feat.bbox[bbox_offset + 0]  * fstride + prior_box_x;
            float cy =          feat.bbox[bbox_offset + 1]  * fstride + prior_box_y;
            float w  = std::exp(feat.bbox[bbox_offset + 2]) * fstride;
            float h  = std::exp(feat.bbox[bbox_offset + 3]) * fstride;
            // clang-format on

            BBox bbox;
            bbox.x_min = cx - w / 2.f;
            bbox.y_min = cy - h / 2.f;
            bbox.x_max = cx + w / 2.f;
            bbox.y_max = cy + h / 2.f;

            // Decode landmarks using prior box
            const size_t kps_offset = offset * KPS_DIM;
            auto point = [&](size_t n) {
                Point p;
                p.x = feat.kps[kps_offset + 2 * n + 0] * fstride + prior_box_x;
                p.y = feat.kps[kps_offset + 2 * n + 1] * fstride + prior_box_y;
                return p;
            };

            // clang-format off
            Landmarks lms;
            lms.right_eye   = point(0);
            lms.left_eye    = point(1);
            lms.nose_tip    = point(2);
            lms.right_mouth = point(3);
            lms.left_mouth  = point(4);

            faces.scores.push_back(score);
            faces.bboxes.push_back(bbox);
            faces.landmarks.push_back(lms);
        }
    }
}

Faces parseOutput(const FeatureMap& feat8, const FeatureMap& feat16, const FeatureMap& feat32,
                  float conf_threshold, float nms_threshold)
{
    Faces found_faces;
    parseBBoxScore(feat8, conf_threshold, found_faces);
    parseBBoxScore(feat16, conf_threshold, found_faces);
    parseBBoxScore(feat32, conf_threshold, found_faces);
    fmt::println("Found {} candidates", found_faces.bboxes.size());

    std::vector<cv::Rect2d> cv_bboxes;
    cv_bboxes.reserve(found_faces.bboxes.size());
    for (const auto& bbox : found_faces.bboxes) {
        cv_bboxes.emplace_back(cv::Point2f(bbox.x_min, bbox.y_min),
                               cv::Point2f(bbox.x_max, bbox.y_max));
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(cv_bboxes, found_faces.scores, conf_threshold,
                      nms_threshold, indices);

    Faces result;

    for (int i : indices) {
        result.bboxes.push_back(found_faces.bboxes[i]);
        result.scores.push_back(found_faces.scores[i]);
        result.landmarks.push_back(found_faces.landmarks[i]);
    }

    fmt::println("Found {} faces", result.bboxes.size());
    return result;
}

cv::Mat readOutputTensorData(const cv::Mat& img, const Resize& resize,
                             const FeatureMap& feat8, const FeatureMap& feat16,
                             const FeatureMap& feat32, float conf_threshold,
                             float nms_threshold)
{
    Faces faces = parseOutput(feat8, feat16, feat32, conf_threshold, nms_threshold);

    float x_scale = 1.f / resize.x_scale;
    float y_scale = 1.f / resize.y_scale;

    auto point = [&](const Point& p) -> cv::Point {
        cv::Point cp;
        cp.x = p.x * x_scale;
        cp.y = p.y * y_scale;
        return cp;
    };

    cv::Mat result = img.clone();

    for (size_t i = 0; i < faces.bboxes.size(); i++) {
        const BBox& bbox = faces.bboxes[i];
        cv::Rect cv_bbox(bbox.x_min * x_scale, bbox.y_min * y_scale,
                         (bbox.x_max - bbox.x_min) * x_scale,
                         (bbox.y_max - bbox.y_min) * y_scale);

        cv::rectangle(result, cv_bbox, cv::Scalar(0, 255, 0), 2);

        cv::putText(result, fmt::format("{:.3f}", faces.scores[i]),
                    cv::Size(bbox.x_min * x_scale, bbox.y_min * y_scale + 12),
                    cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255));

        const Landmarks& lms = faces.landmarks[i];
        cv::circle(result, point(lms.right_eye), 4, cv::Scalar(255, 0, 0), -1);
        cv::circle(result, point(lms.left_eye), 4, cv::Scalar(0, 0, 255), -1);
        cv::circle(result, point(lms.nose_tip), 4, cv::Scalar(0, 255, 0), -1);
        cv::circle(result, point(lms.right_mouth), 4, cv::Scalar(255, 0, 255), -1);
        cv::circle(result, point(lms.left_mouth), 4, cv::Scalar(0, 255, 255), -1);
    }

    return result;
}

class Demo {
public:
    void init(std::string_view model_path);
    cv::Mat run(const cv::Mat& img, int tensor_size,
                float conf_threshold, float nms_threshold);

private:
    Ort::Env m_env{nullptr};
    Ort::Session m_session{nullptr};
    Ort::MemoryInfo m_mem_info{nullptr};
};

}  // namespace

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
        constexpr int device_id = 0;  // Might not work if you have multiple GPUs
        m_mem_info = Ort::MemoryInfo("Cuda", OrtArenaAllocator, device_id, OrtMemTypeDefault);
    } else {
        m_mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    }
}

cv::Mat Demo::run(const cv::Mat& img, int tensor_size,
                  float conf_threshold, float nms_threshold)
{
    Resize resize = computeResize(img.size().width, img.size().height, tensor_size);

    fmt::println("Resizing input image from {}x{} to {}x{} (scale = [{:.5}, {:.5}]) {}x{}",
                 resize.img_w, resize.img_h, resize.new_w, resize.new_h,
                 resize.x_scale, resize.y_scale, resize.tensor_w, resize.tensor_h);

    // Vectors initialize to zero which is unnecessary for our use cases.
    // Consider using a structure with different construct semantics.
    std::vector<float> input = generateInputTensorData(img, resize);
    auto input_shape = inputShape(resize.tensor_w, resize.tensor_h);

    Ort::Value input_tensor = Ort::Value::CreateTensor(
        m_mem_info, input.data(), input.size(), input_shape.data(), input_shape.size());

    constexpr int64_t NUM_INPUTS = 1;
    constexpr std::array<const char*, NUM_INPUTS> INPUT_NAMES{"input"};

    constexpr int64_t NUM_OUTPUTS = 12;
    // This needs to be kept in sync with generateFeatureMap()
    constexpr std::array<const char*, NUM_OUTPUTS> OUTPUT_NAMES{
        "cls_8", "obj_8", "bbox_8", "kps_8",
        "cls_16", "obj_16", "bbox_16", "kps_16",
        "cls_32", "obj_32", "bbox_32", "kps_32"
    };
    std::vector<Ort::Value> output_tensors;
    output_tensors.reserve(NUM_OUTPUTS);

    FeatureMap feat8 = generateFeatureMap(8, output_tensors, m_mem_info, resize);
    FeatureMap feat16 = generateFeatureMap(16, output_tensors, m_mem_info, resize);
    FeatureMap feat32 = generateFeatureMap(32, output_tensors, m_mem_info, resize);

    Ort::RunOptions run_options;
    m_session.Run(run_options, INPUT_NAMES.data(), &input_tensor, NUM_INPUTS,
                  OUTPUT_NAMES.data(), output_tensors.data(), NUM_OUTPUTS);

    return readOutputTensorData(img, resize, feat8, feat16, feat32,
                                conf_threshold, nms_threshold);
}

int main(int argc, char *argv[])
{
    if (argc != 6 && argc != 7) {
        printHelp();
        return -1;
    }

    if (std::string_view(argv[1]) == "--help") {
        printHelp();
        return 0;
    }

    const std::string_view model_path = argv[1];
    int tensor_size = std::stoi(argv[2]);
    float conf_threshold = std::stof(argv[3]);
    float nms_threshold = std::stof(argv[4]);
    const std::string image_path = argv[5];

    const std::string out_img_path = (argc == 7) ? argv[6] : "";
    if (!out_img_path.empty()) {
        fmt::println("Annotated image will be written to: {}", out_img_path);
    }

    const cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR_BGR);
    cv::Mat display;

    // Have the Ort::Session lifetime outlive the first waitKey(0) so it is
    // easier to check memory usage of the model.
    Demo demo;

    try {
        fmt::println("Setting up model...");
        demo.init(model_path);
        fmt::println("Done!");
        fmt::println("Running inference...");
        display = demo.run(img, tensor_size, conf_threshold, nms_threshold);
        fmt::println("Done!");
    } catch (const std::exception& e) {
        fmt::println("{}", e.what());
        return -1;
    }

    if (!out_img_path.empty()) {
        fmt::println("Writing detections: {}", out_img_path);
        cv::imwrite(out_img_path, display);
    }

    const std::string WINDOW = "YuNet Demo";
    cv::namedWindow(WINDOW, cv::WINDOW_NORMAL);
    cv::resizeWindow(WINDOW, WINDOW_WIDTH, WINDOW_HEIGHT);

    cv::imshow(WINDOW, display);

    cv::waitKey(0);

    cv::destroyAllWindows();
    return 0;
}
