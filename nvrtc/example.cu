#include <cstdlib>
#include <string>
#include <vector>
#include <memory>
#include <cassert>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

[[noreturn]] void fail(const std::string& msg, int code) {
    std::cerr << "error: " << msg << " (" << code << ')' << std::endl;
    std::exit(EXIT_FAILURE);
}


std::unique_ptr<char[]> compile_to_ptx(const char* program_source) {
    nvrtcResult rv;

    // create nvrtc program
    nvrtcProgram prog;
    rv = nvrtcCreateProgram(
        &prog,
        program_source,
        "program.cu",
        0,
        nullptr,
        nullptr
    );
    if(rv != NVRTC_SUCCESS) fail("nvrtcCreateProgram", rv);
    printf("ok\n");
    // compile nvrtc program
    
    //options.push_back("-default-device");
    rv = nvrtcCompileProgram(prog, 0, nullptr);
    if(rv != NVRTC_SUCCESS) {
        std::size_t log_size;
        rv = nvrtcGetProgramLogSize(prog, &log_size);
        if(rv != NVRTC_SUCCESS) fail("nvrtcGetProgramLogSize", rv);

        auto log = std::make_unique<char[]>(log_size);
        rv = nvrtcGetProgramLog(prog, log.get());
        if(rv != NVRTC_SUCCESS) fail("nvrtcGetProgramLog", rv);
        assert(log[log_size - 1] == '\0');

        std::cerr << "Compile error; log:\n" << log.get() << std::endl;

        fail("nvrtcCompileProgram", rv);
    }

    // get ptx code
    std::size_t ptx_size;
    rv = nvrtcGetPTXSize(prog, &ptx_size);
    if(rv != NVRTC_SUCCESS) fail("nvrtcGetPTXSize", rv);

    auto ptx = std::make_unique<char[]>(ptx_size);
    rv = nvrtcGetPTX(prog, ptx.get());
    if(rv != NVRTC_SUCCESS) fail("nvrtcGetPTX", rv);
    assert(ptx[ptx_size - 1] == '\0');

    nvrtcDestroyProgram(&prog);

    return ptx;
}

const char program_source[] = R"%%%(
//#include <stdint.h>
extern "C" __global__ void f(int* in, int* out) {
    out[threadIdx.x] = in[threadIdx.x];
}
)%%%";

int main() {
    CUresult rv;

    // initialize CUDA
    rv = cuInit(0);
    if(rv != CUDA_SUCCESS) fail("cuInit", rv);

    // compile program to ptx
    auto ptx = compile_to_ptx(program_source);
    std::cout << "PTX code:\n" << ptx.get() << std::endl;
}