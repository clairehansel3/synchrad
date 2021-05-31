cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)
project(synchrad VERSION 0.0.0 LANGUAGES CXX)

add_library(synchrad SHARED
  src/synchrad.cxx
)

set_target_properties(synchrad PROPERTIES CXX_EXTENSIONS OFF CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
target_compile_options(synchrad PRIVATE -O3 -funsafe-math-optimizations)
install(TARGETS synchrad DESTINATION lib)
