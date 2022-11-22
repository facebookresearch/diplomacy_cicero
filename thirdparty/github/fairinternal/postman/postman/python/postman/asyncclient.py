#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import postman.rpc


class AsyncClient(postman.rpc.AsyncClient):
    def connect(self, deadline_sec=60):
        return Streams(super(AsyncClient, self).connect(deadline_sec))


class Streams:
    def __init__(self, raw_stream):
        self._raw_stream = raw_stream

    # TODO(heiner): Consider implementing this on the C++ side.
    def __getattr__(self, name):
        return lambda *args: self._raw_stream.call(name, args)

    def close(self):
        return self._raw_stream.close()
