// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2025 Daniel Gao

#pragma once

#include "cpputil.h"

#include "fmt/format.h"
#include "fmt/ranges.h"

#include <onnxruntime_cxx_api.h>

#include <string>
#include <string_view>
#include <vector>

const char* to_cstr(ONNXTensorElementDataType type);

struct NodeInfo {
    std::string name;
    std::vector<int64_t> dims;
    ONNXTensorElementDataType type;
};

std::vector<NodeInfo> findInputNodeInfo(const Ort::Session& session);
std::vector<NodeInfo> findOutputNodeInfo(const Ort::Session& session);

template <class LogFunc>
void printNodeInfo(LogFunc log, std::string_view title,
                   const std::vector<NodeInfo>& nodes)
{
    std::invoke(log, title);
    for (const auto& node : nodes) {
        std::invoke(log, fmt::format("  Name: {}", node.name));
        std::invoke(log, fmt::format("    Dim: [{}]", fmt::join(node.dims, ", ")));
        std::invoke(log, fmt::format("    Type: {}, ({})", to_cstr(node.type),
                                     rnx::to_underlying(node.type)));
    }
}
