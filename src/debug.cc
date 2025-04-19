// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2025 Daniel Gao

#include "debug.h"

const char* to_cstr(ONNXTensorElementDataType type)
{
    // Neither an exhaustive nor ordered list of all possible enums.
    static std::unordered_map<ONNXTensorElementDataType, const char*> mapping = {
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED, "Undefined"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, "f16"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "f32"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, "f64"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4, "u4"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, "u8"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, "u16"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32, "u32"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64, "u64"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4, "i4"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, "i8"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, "i16"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, "i32"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, "i64"},
        {ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, "bool"}};

    auto it = mapping.find(type);
    return it == mapping.end() ? "Unknown" : it->second;
}

std::vector<NodeInfo> findInputNodeInfo(const Ort::Session& session)
{
    std::vector<NodeInfo> result;

    const size_t num_input_nodes = session.GetInputCount();
    result.reserve(num_input_nodes);

    const auto memory_info =
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    const Ort::Allocator allocator(session, memory_info);

    for (size_t i = 0; i < num_input_nodes; i++) {
        NodeInfo info;

        const auto name_ptr = session.GetInputNameAllocated(i, allocator);
        info.name = std::string(name_ptr.get());

        const Ort::TypeInfo type_info(session.GetInputTypeInfo(i));
        const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        info.dims = tensor_info.GetShape();
        info.type = tensor_info.GetElementType();

        result.push_back(info);
    }

    return result;
}

std::vector<NodeInfo> findOutputNodeInfo(const Ort::Session& session)
{
    std::vector<NodeInfo> result;

    const size_t num_output_nodes = session.GetOutputCount();
    result.reserve(num_output_nodes);

    const auto memory_info =
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    const Ort::Allocator allocator(session, memory_info);

    for (size_t i = 0; i < num_output_nodes; i++) {
        NodeInfo info;

        const auto name_ptr = session.GetOutputNameAllocated(i, allocator);
        info.name = std::string(name_ptr.get());

        const Ort::TypeInfo type_info(session.GetOutputTypeInfo(i));
        const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        info.dims = tensor_info.GetShape();
        info.type = tensor_info.GetElementType();

        result.push_back(info);
    }

    return result;
}
