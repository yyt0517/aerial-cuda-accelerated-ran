# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

get_filename_component(ProjectId ${CMAKE_CURRENT_SOURCE_DIR} NAME)
add_executable(${ProjectId})
target_sources(${ProjectId} PRIVATE main.cpp trafficGenerator.cpp)
target_include_directories(${ProjectId} PRIVATE ${HDF5_CXX_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/../src/)
target_link_libraries(${ProjectId} PRIVATE cumac ${HDF5_CXX_LIBRARIES})

if (ENABLE_TESTS)
    add_executable(tbsUnitsTest tbsUnitsTest.cpp)
    add_test(NAME tbsUnitsTest COMMAND tbsUnitsTest)
    add_executable(trafficGranularityProfileTest trafficGranularityProfileTest.cpp)
    add_test(NAME trafficGranularityProfileTest COMMAND trafficGranularityProfileTest)

    add_executable(trafficPacketAccountingTest)
    target_sources(trafficPacketAccountingTest PRIVATE trafficPacketAccountingTest.cpp trafficGenerator.cpp)
    target_include_directories(
        trafficPacketAccountingTest
        PRIVATE ${HDF5_CXX_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/../src/)
    target_link_libraries(trafficPacketAccountingTest PRIVATE cumac ${HDF5_CXX_LIBRARIES})
    add_test(NAME trafficPacketAccountingTest COMMAND trafficPacketAccountingTest)
endif()
