add_executable(
        GenerateImg
        GenerateImg.cpp)

target_link_libraries(GenerateImg ${OpenCL_LIBRARY} ${OpenCV_LIBS})

