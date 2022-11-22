/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include "../cc/message.h"

namespace py = pybind11;

namespace dipcc {

py::dict py_message_to_dict(const Message &message);

py::dict py_messages_to_phase_dict(const std::map<uint64_t, Message> &messages);

py::dict py_message_history_to_dict(
    const std::map<Phase, std::map<uint64_t, Message>> &message_history,
    Phase exclude_phase);

}; // namespace dipcc
