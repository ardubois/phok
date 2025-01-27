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
        nullptr,
        0,
        nullptr,
        nullptr
    );
    if(rv != NVRTC_SUCCESS) fail("nvrtcCreateProgram", rv);
    printf("ok\n");
    // compile nvrtc program
    
    //options.push_back("-default-device");
    std::vector<const char*> options = {
        "--include-path=/lib/erlang/usr/include/",
        "--include-path=/usr/include"
    };
    rv = nvrtcCompileProgram(prog, options.size(), options.data());
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

const char program2[] = R"%%%(
#include "erl_nif.h"


__device__
int anon_45cf36d0dd(int x)
{
return ((x + 1));
}


__device__
int cc(int a)
{
return ((a + a));
}


__device__
int g(int a)
{
return (cc((a + a)));
}


__global__
void map_ske(int *a1, int *a2, int size)
{
int id = ((blockIdx.x * blockDim.x) + threadIdx.x);
int r = g(a1[id]);
if((id < size))
{
	a2[id] = anon_45cf36d0dd(a1[id]);
}

}

extern "C" void map_ske_call(ErlNifEnv *env, const ERL_NIF_TERM argv[], ErlNifResourceType* type,ErlNifResourceType* ftype)
  {

    ERL_NIF_TERM list;
    ERL_NIF_TERM head;
    ERL_NIF_TERM tail;

   // void **fun_res;

    const ERL_NIF_TERM *tuple_blocks;
    const ERL_NIF_TERM *tuple_threads;
    int arity;

    if (!enif_get_tuple(env, argv[1], &arity, &tuple_blocks)) {
      printf ("spawn: blocks argument is not a tuple");
    }

    if (!enif_get_tuple(env, argv[2], &arity, &tuple_threads)) {
      printf ("spawn:threads argument is not a tuple");
    }
    int b1,b2,b3,t1,t2,t3;

    enif_get_int(env,tuple_blocks[0],&b1);
    enif_get_int(env,tuple_blocks[1],&b2);
    enif_get_int(env,tuple_blocks[2],&b3);
    enif_get_int(env,tuple_threads[0],&t1);
    enif_get_int(env,tuple_threads[1],&t2);
    enif_get_int(env,tuple_threads[2],&t3);

    dim3 blocks(b1,b2,b3);
    dim3 threads(t1,t2,t3);

    list= argv[3];

  int **array_res1;
    enif_get_list_cell(env,list,&head,&tail);
    enif_get_resource(env, head, type, (void **) &array_res1);
    int *arg1 = *array_res1;
    list = tail;

    int **array_res2;
    enif_get_list_cell(env,list,&head,&tail);
    enif_get_resource(env, head, type, (void **) &array_res2);
    int *arg2 = *array_res2;
    list = tail;

    enif_get_list_cell(env,list,&head,&tail);
  int arg3;
  enif_get_int(env, head, &arg3);
  list = tail;

   map_ske<<<blocks, threads>>>(arg1,arg2,arg3);
    cudaError_t error_gpu = cudaGetLastError();
    if(error_gpu != cudaSuccess)
     { char message[200];
       strcpy(message,"Error kernel call: ");
       strcat(message, cudaGetErrorString(error_gpu));
       enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
     }
}
)%%%";

int main() {
    CUresult rv;

    // initialize CUDA
    rv = cuInit(0);
    if(rv != CUDA_SUCCESS) fail("cuInit", rv);
    printf("inicio\n");
    // compile program to ptx
    auto ptx = compile_to_ptx(program2);
    std::cout << "PTX code:\n" << ptx.get() << std::endl;
}