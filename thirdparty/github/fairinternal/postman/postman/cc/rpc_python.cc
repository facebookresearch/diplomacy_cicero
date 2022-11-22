/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/

#include <optional>

// Enable including numpy via numpy_stub.h.
#define USE_NUMPY 1
#include <torch/all.h>
#include <torch/python.h>

#include <nest.h>
#include <nest_pybind.h>

#include "postman/asyncclient.h"
#include "postman/client.h"
#include "postman/computationqueue.h"
#include "postman/exceptions.h"
#include "postman/server.h"

namespace py = pybind11;

using namespace postman;

template <typename Function>
std::invoke_result_t<Function> callwithcatch(Function f) {
  try {
    py::gil_scoped_release release;
    return f();
  } catch (const CallError &e) {
    std::string message = e.what();
    std::size_t pos = message.find(": ");
    if (pos != std::string::npos) {
      std::string type = message.substr(0, pos);
      py::object builtins = py::module::import("builtins");
      if (py::hasattr(builtins, type.c_str())) {
        py::object error_type = builtins.attr(type.c_str());
        if (PyExceptionClass_Check(error_type.ptr())) {
          message = message.substr(pos + 2);
          PyErr_SetString(error_type.ptr(), message.c_str());
          throw py::error_already_set();
        }
      }
    }
    throw;
  }
}

PYBIND11_MODULE(rpc, m) {
  py::register_exception_translator([](std::exception_ptr ptr) {
    try {
      if (ptr) std::rethrow_exception(ptr);
    } catch (const QueueClosed &e) {
      PyErr_SetString(PyExc_StopIteration, e.what());
    } catch (const ConnectionError &e) {
      PyErr_SetString(PyExc_ConnectionError, e.what());
    } catch (const TimeoutError &e) {
      PyErr_SetString(PyExc_ConnectionError, e.what());
    } catch (const CallError &e) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
    }
  });

  // Clients.

  py::class_<Client>(m, "Client")
      .def(py::init<const std::string &>(), py::arg("address"))
      .def("connect", &Client::connect, py::arg("deadline_sec") = 60)
      .def("call",
           [](Client *client, const std::string &function,
              const TensorNest &inputs) -> TensorNest {
             // TODO(heiner): This Python obj -> TensorNest incurs overhead,
             // and is unnecessary anyway given that we just want to serialize.
             // Consider removing all of nest here and just dealing with lists
             // of tensors, or "named" tensors (pairs of strings and tensors).
             return callwithcatch(
                 [&]() { return client->call(function, inputs); });
           });

  py::class_<std::future<TensorNest>>(m, "PostmanFuture")
      .def("get", &std::future<TensorNest>::get,
           py::call_guard<py::gil_scoped_release>())
      .def("wait", &std::future<TensorNest>::wait,
           py::call_guard<py::gil_scoped_release>());

  py::class_<AsyncClient::Streams, std::shared_ptr<AsyncClient::Streams>>(
      m, "Streams")
      .def("call",
           [](AsyncClient::Streams *self, const std::string &function,
              const TensorNest &inputs) -> std::future<TensorNest> {
             // TODO(heiner): As above.
             return callwithcatch(
                 [&]() { return self->call(function, inputs); });
           })
      .def("close", &AsyncClient::Streams::close,
           py::call_guard<py::gil_scoped_release>());

  py::class_<AsyncClient>(m, "AsyncClient")
      .def(py::init<const std::string &>(), py::arg("address"))
      .def("connect", &AsyncClient::connect, py::arg("deadline_sec") = 60);

  // Server.

  py::class_<Server>(m, "Server")
      .def(py::init<const std::string &>(), py::arg("address"))
      .def("run", &Server::run)
      .def("running", &Server::running)
      .def("port", &Server::port)
      .def("wait", &Server::wait, py::call_guard<py::gil_scoped_release>())
      .def("stop", &Server::stop, py::call_guard<py::gil_scoped_release>())
      .def("bind_queue", &Server::bind_queue, py::arg("name"), py::arg("queue"))
      .def("bind_queue_batched", &Server::bind_queue_batched, py::arg("name"),
           py::arg("queue"));

  py::class_<ComputationQueue::Computation,
             std::shared_ptr<ComputationQueue::Computation>>(m, "Computation")
      .def("get_inputs", &ComputationQueue::Computation::get_inputs)
      .def(
          "set_outputs",
          [](std::shared_ptr<ComputationQueue::Computation> self,
             std::optional<TensorNest> outputs) {
            self->set_outputs(std::move(outputs).value_or(
                TensorNest(std::vector<TensorNest>())));
          },
          py::arg("outputs"))
      .def("__enter__",
           [](std::shared_ptr<ComputationQueue::Computation> self) {
             return self;
           })
      .def("__exit__", [](std::shared_ptr<ComputationQueue::Computation> self,
                          py::object type, py::object value, py::object trace) {
        if (type.is_none()) return false;

        // We resolve the promise with an exception here, but not
        // py::error_already_set, which would require the GIL when
        // handling the associated future.
        std::string message = type.attr("__name__").cast<std::string>() + ": " +
                              py::str(value).cast<std::string>();

        py::object pstream = py::module::import("io").attr("StringIO")();
        PyTraceBack_Print(trace.ptr(), pstream.ptr());
        message += "\n\n" + pstream.attr("getvalue")().cast<std::string>();

        self->set_exception(
            std::make_exception_ptr(std::runtime_error(message)));
        return true;
      });

  py::class_<ComputationQueue, std::shared_ptr<ComputationQueue>>(
      m, "ComputationQueue")
      .def(py::init<uint32_t>(), py::arg("batch_size"))
      .def("close", &ComputationQueue::close)
      .def("__iter__",
           [](std::shared_ptr<ComputationQueue> self) { return self; })
      .def("__next__", &ComputationQueue::get,
           py::arg("wait_till_full") = false,
           py::call_guard<py::gil_scoped_release>())
      .def("get", &ComputationQueue::get, py::arg("wait_till_full") = false,
           py::call_guard<py::gil_scoped_release>())
      .def("set_batch_size", &ComputationQueue::set_batch_size,
           py::arg("batch_size"));
}
