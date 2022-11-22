#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import List
import atexit
import glob
import os
import pathlib
import subprocess

from setuptools import setup, find_packages
from setuptools.command.install import install


def _post_install():
    # Compiling the schema
    subprocess.check_output(
        ["protoc"] + list(glob.glob("conf/*.proto")) + ["--python_out", "./", "--mypy_out", "./"]
    )


def _read_requirements() -> List[str]:
    requirements = []
    with (pathlib.Path(__file__).parent / "requirements.txt").open() as stream:
        for line in stream:
            line = line.split("#")[0].strip()
            if line:
                requirements.append(line)
    return requirements


class PostInstallBoilerplate(install):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        atexit.register(_post_install)


setup(
    name="fairdiplomacy",
    version="0.1",
    packages=find_packages(),
    install_requires=_read_requirements(),
    entry_points={
        "console_scripts": [
            "diplom=parlai_diplomacy.scripts.diplom:main",
            "parlai-sweep=parlai_diplomacy.utils.param_sweeps.collector:main",
        ]
    },
    cmdclass={"install": PostInstallBoilerplate},
)
