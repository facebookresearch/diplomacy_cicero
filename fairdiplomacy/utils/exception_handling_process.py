#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import traceback
import signal

from fairdiplomacy.utils.multiprocessing_spawn_context import get_multiprocessing_ctx

mp = get_multiprocessing_ctx()

# mostly from https://stackoverflow.com/questions/19924104/python-multiprocessing-handling-child-errors-in-parent
class ExceptionHandlingProcess(mp.Process):
    """A process that propagates exceptions back to the parent on join()."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None

    def run(self):
        try:
            signal.signal(signal.SIGINT, self.signal_handler)
            # signal.signal(signal.SIGSEGV, self.signal_handler)

            super().run()
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))
            raise e

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception

    def join(self):
        super().join()
        if self.exception:
            error, traceback = self.exception
            print(traceback)
            raise error

    def signal_handler(self, signal, frame):
        print("Caught signal: ", signal, frame)
