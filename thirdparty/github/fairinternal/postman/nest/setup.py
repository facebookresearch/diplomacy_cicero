#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
#
#   CXX=c++ python3 setup.py build develop
# or
#   CXX=c++ pip install . -vv
#

import sys

import setuptools
import setuptools.command.build_ext


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11

        return pybind11.get_include(self.user)


ext_modules = [
    setuptools.Extension(
        "nest._nest",
        ["nest/nest_pybind.cc"],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
        ],
        depends=["nest/nest.h", "nest/nest_pybind.h"],
        language="c++",
        extra_compile_args=["-std=c++17"],
    )
]


class BuildExt(setuptools.command.build_ext.build_ext):
    """A custom build extension for adding compiler-specific options."""

    c_opts = {"msvc": ["/EHsc"], "unix": []}

    if sys.platform == "darwin":
        c_opts["unix"] += ["-stdlib=libc++", "-mmacosx-version-min=10.14"]

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == "unix":
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append("-std=c++17")
            opts.append("-fvisibility=hidden")
        elif ct == "msvc":
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args += opts
            if sys.platform == "darwin":
                ext.extra_link_args = ["-stdlib=libc++"]

        super().build_extensions()


setuptools.setup(
    name="nest",
    packages=["nest"],
    version="0.0.6",
    author="Heinrich Kuttler",
    author_email="hnr@fb.com",
    ext_modules=ext_modules,
    headers=["nest/nest.h", "nest/nest_pybind.h"],
    cmdclass={"build_ext": BuildExt},
    install_requires=["pybind11>=2.4"],
    setup_requires=["pybind11>=2.4"],
)
