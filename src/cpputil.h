// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2025 Daniel Gao

#pragma once

#include <onnxruntime_cxx_api.h>

#include <array>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace rnx {

template <class Enum>
constexpr std::underlying_type_t<Enum> to_underlying(Enum e) noexcept
{
    return static_cast<std::underlying_type_t<Enum>>(e);
}

inline bool startsWith(std::string_view str, std::string_view prefix)
{
    return std::string_view(str.data(), std::min(str.size(), prefix.size())) == prefix;
}

}  // namespace rnx
