set(TARGET llama-share-prewarm)
add_executable(${TARGET} llama-share-prewarm.cpp)
# llama-share.h holds the registry wire format; the program drives the ggml public
# API directly (buffer types, tensor set, gguf reader). The CUDA backend is loaded
# at runtime (ggml_backend_load_all / GGML_BACKEND_PATH), not linked.
target_include_directories(${TARGET} PRIVATE ../../src ../../ggml/include)
target_link_libraries(${TARGET} PRIVATE ggml ggml-cpu)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()