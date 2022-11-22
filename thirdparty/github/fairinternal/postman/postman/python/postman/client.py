#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import postman.rpc


class Client(postman.rpc.Client):
    # TODO(heiner): Consider implementing this on the C++ side.
    def __getattr__(self, name):
        return lambda *args: self.call(name, args)
