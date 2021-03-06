cmake_minimum_required(VERSION 3.9)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# decide on subdirectory name within native_libs
if(WIN32)
    set(PLATFORM_DIR_SUFFIX windows)
elseif(APPLE)
    set(PLATFORM_DIR_SUFFIX macos)
else()
    set(PLATFORM_DIR_SUFFIX linux)
endif()

# set output directory
set(OUT_DIR ${CMAKE_CURRENT_LIST_DIR}/../${PLATFORM_DIR_SUFFIX})
if(WIN32)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUT_DIR})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${OUT_DIR})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${OUT_DIR})
else()
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${OUT_DIR})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${OUT_DIR})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${OUT_DIR})
endif()

set(CMAKE_CXX_VISIBILITY_PRESET hidden)

project(TFL)

find_package(ZLIB REQUIRED)
find_package(PNG REQUIRED)

# Includes path: project root
include_directories(${PNG_INCLUDE_DIR})

if(WIN32 OR DEFINED ENV{TF_USE_LOCAL_LIBRARY})
    include_directories(${CMAKE_SOURCE_DIR}/../../../../tensorflow/include/)
    link_directories(${CMAKE_SOURCE_DIR}/../../../../tensorflow/lib)
    file(GLOB LIBRARIES_TO_COPY
      "${CMAKE_SOURCE_DIR}/../../../../tensorflow/lib/*"
    )
    file(COPY ${LIBRARIES_TO_COPY} DESTINATION ${OUT_DIR})
endif()

# Glob for the project files
file(GLOB_RECURSE HEADER_FILES ${PROJECT_SOURCE_DIR}/*.h)
file(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/*.cpp)

# Define the compilation target
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SRC_FILES})
target_compile_definitions(${PROJECT_NAME} PRIVATE BUILDING_TFL_HELPER)
#target_compile_definitions(${PROJECT_NAME} PRIVATE VERBOSE) # for debugging
#target_compile_definitions(${PROJECT_NAME} PRIVATE VERBOSEFFI) # for debugging
#target_compile_definitions(${PROJECT_NAME} PRIVATE LOG_GRAPH_STRUCTURE) # for debugging
target_compile_definitions(${PROJECT_NAME} PUBLIC)

# set warning level
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W4)
else(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra)
endif(MSVC)

target_include_directories(${PROJECT_NAME} PUBLIC ../../../..)

target_link_libraries(${PROJECT_NAME} tensorflow ${PNG_LIBRARY} ${ZLIB_LIBRARY})

###################
