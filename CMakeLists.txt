cmake_minimum_required(VERSION 3.2)

SET (SURF_LIB_SOURCES
        fasthessian
        ImagenIntegral
        surf
        utils
        ipoint
)

SET (SURF_FEAT_SOURCES main)

find_package(OpenCV REQUIRED)

add_library(surf ${SURF_LIB_SOURCES})


add_executable(surffeat ${SURF_FEAT_SOURCES})
target_link_libraries(surffeat surf ${OpenCV_LIBS})