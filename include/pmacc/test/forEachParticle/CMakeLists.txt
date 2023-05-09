# Copyright 2016-2023 Alexander Grund, Bernhard Manfred Gruber
#
# This file is part of PMacc.
#
# PMacc is free software: you can redistribute it and/or modify
# it under the terms of either the GNU General Public License or
# the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PMacc is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License and the GNU Lesser General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License
# and the GNU Lesser General Public License along with PMacc.
# If not, see <http://www.gnu.org/licenses/>.
#

cmake_minimum_required(VERSION 3.18.0)
project(TestForEachParticle)

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/../..")

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)

if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()

find_package(PMacc REQUIRED CONFIG PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../..")
cupla_add_executable(${PROJECT_NAME} forEachParticleUT.cpp)
#target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_link_libraries(${PROJECT_NAME} PRIVATE pmacc::pmacc)

add_custom_target(run
    COMMAND mpiexec -n 1 TestRng
    DEPENDS TestRng
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
