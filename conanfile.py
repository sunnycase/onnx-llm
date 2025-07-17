# Copyright 2019-2021 Canaan Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

from conan import ConanFile
from conan.errors import ConanInvalidConfiguration
from conan.tools.build import check_min_cppstd, cross_building
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.files import apply_conandata_patches, copy, export_conandata_patches, get, rmdir


class onnxllmConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "tests": [True, False],
        "nncase_dir": ["ANY"]
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "tests": False,
        "nncase_dir": ""
    }

    @property
    def _min_cppstd(self):
        return 20

    def layout(self):
        cmake_layout(self)

    def requirements(self):
        self.requires('nlohmann_json/3.9.1')

    def build_requirements(self):
        pass

    def configure(self):
        pass
        
    def validate(self):
        if self.settings.compiler.get_safe("cppstd"):
            check_min_cppstd(self, self._min_cppstd)

    def generate(self):
        tc = CMakeToolchain(self, generator="Ninja")
        tc.variables['BUILD_TESTING'] = self.options.tests
        tc.variables['nncase_DIR'] = "/mnt/home-nas/work/repo/nncase-v3/nncase/install/lib/cmake/nncase"
        tc.generate()
        deps = CMakeDeps(self)
        deps.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
