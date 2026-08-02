cmake_minimum_required(VERSION 3.24)
project(cudarepo_extension LANGUAGES CXX CUDA)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")

find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(_C MODULE
    csrc/fused_bias_relu.cpp
    csrc/fused_bias_relu_cuda.cu
)
target_link_libraries(_C PRIVATE ${TORCH_LIBRARIES} Python3::Python CUDA::cudart)
target_compile_features(_C PRIVATE cxx_std_17)
target_compile_definitions(_C PRIVATE TORCH_EXTENSION_NAME=_C)
if(MSVC)
    target_compile_options(_C PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/O2>)
else()
    target_compile_options(_C PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
endif()
target_compile_options(_C PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>)
set_target_properties(_C PROPERTIES
    PREFIX ""
    POSITION_INDEPENDENT_CODE ON
    CUDA_ARCHITECTURES native
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/cudarepo"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/cudarepo"
)
if(WIN32)
    set_target_properties(_C PROPERTIES SUFFIX ".pyd" WINDOWS_EXPORT_ALL_SYMBOLS ON)
    foreach(CONFIG Debug Release RelWithDebInfo MinSizeRel)
        string(TOUPPER "${CONFIG}" CONFIG_UPPER)
        set_target_properties(_C PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${CMAKE_CURRENT_SOURCE_DIR}/src/cudarepo"
        )
    endforeach()
endif()
