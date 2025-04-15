// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2025 Daniel Gao

#include "src/debug.h"

#include "fmt/format.h"

#include <onnxruntime_cxx_api.h>

int main(int argc, char** argv) noexcept(false)
{
    if (argc < 2) {
        fmt::println("Usage: onnx-dump <model.onnx>");
        return -1;
    }
    const char* model_path = argv[1];

    const Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "onnx-dump");
    const Ort::SessionOptions session_options;
    const Ort::Session session(env, model_path, session_options);

    fmt::println("Dumping model: {}", model_path);

    auto log = [](std::string_view sv) { fmt::println("{}", sv); };

    const auto input_info = findInputNodeInfo(session);
    printNodeInfo(log, "Input Nodes:", input_info);
    const auto output_info = findOutputNodeInfo(session);
    printNodeInfo(log, "Output Nodes:", output_info);

    return 0;
}
