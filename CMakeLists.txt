cmake_minimum_required(VERSION 3.12)

# set the project name
project(MueLuExample)

add_executable(MueLuExample mueluexample.cc)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
target_link_libraries(MueLuExample ${CUDA_LIBRARIES})

find_package(Kokkos REQUIRED)
include_directories(${Kokkos_INCLUDE_DIRS})
target_link_libraries(MueLuExample kokkos)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
target_link_libraries (MueLuExample ${MPI_LIBRARIES})

find_package(Trilinos REQUIRED)
include_directories(${Trilinos_INCLUDE_DIRS})
target_link_libraries (MueLuExample ${Trilinos_LIBRARIES})

find_package(MueLu REQUIRED)
include_directories(${MueLu_INCLUDE_DIRS})
target_link_libraries (MueLuExample ${MueLu_LIBRARIES})

find_package(Teuchos REQUIRED)
include_directories(${Teuchos_INCLUDE_DIRS})
target_link_libraries (MueLuExample ${Teuchos_LIBRARIES})

