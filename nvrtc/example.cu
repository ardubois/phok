#include "erl_nif.h"
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



char* compile_to_ptx(const char* program_source) {
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
    int size_options = 10;
     const char* options[10] = {
        "--include-path=/lib/erlang/usr/include/",
        "--include-path=/usr/include/",
        "--include-path=/usr/lib/",
        "--include-path=/usr/include/x86_64-linux-gnu/",
        "--include-path=/usr/include/c++/11",
        "--include-path=/usr/include/x86_64-linux-gnu/c++/11",
        "--include-path=/usr/include/c++/11/backward",
        "--include-path=/usr/lib/gcc/x86_64-linux-gnu/11/include",
        "--include-path=/usr/include/i386-linux-gnu/",
        "--include-path=/usr/local/include"
 };
    rv = nvrtcCompileProgram(prog, size_options, options);
    if(rv != NVRTC_SUCCESS) {
        size_t log_size;
        rv = nvrtcGetProgramLogSize(prog, &log_size);
        if(rv != NVRTC_SUCCESS) fail("nvrtcGetProgramLogSize", rv);

        //auto log = std::make_unique<char[]>(log_size);
        char log[10];
        rv = nvrtcGetProgramLog(prog, log);
        if(rv != NVRTC_SUCCESS) fail("nvrtcGetProgramLog", rv);
        assert(log[log_size - 1] == '\0');

        printf("Compile error; log: %s\n", log);

        fail("nvrtcCompileProgram", rv);
    }

    // get ptx code
    size_t ptx_size;
    rv = nvrtcGetPTXSize(prog, &ptx_size);
    if(rv != NVRTC_SUCCESS) fail("nvrtcGetPTXSize", rv);
    char* ptx_source = new char[ptx_size];
    nvrtcGetPTX(prog, ptx_source);
  
   
    if(rv != NVRTC_SUCCESS) fail("nvrtcGetPTX", rv);
    assert(ptx_source[ptx_size - 1] == '\0');

    nvrtcDestroyProgram(&prog);

    return ptx_source;
}


const char program2[] = R"%%%(

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


extern "C" __global__
 void map_ske(int *a1, int *a2, int size)
{
int id = ((blockIdx.x * blockDim.x) + threadIdx.x);
int r = g(1);
if((id < size))
{
	a2[id] = anon_45cf36d0dd(a1[id]) + r +cc(1);
  
}

}


)%%%";

int main() {
    CUresult err;
    CUdevice   device;
    CUcontext  context;
    CUmodule   module;
    CUfunction function;
    char       *kernel_name = (char*) "map_ske";

    // initialize CUDA
    err = cuInit(0);
    
    if(err != CUDA_SUCCESS)  
      { char message[200];
        const char *error;
        cuGetErrorString(err, &error);
        strcpy(message,"Error create_ref_nif: ");
        strcat(message, error);
        printf("%s\n",error);
        exit(-1);
        //enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
      }

    

    // compile program to ptx
    char* ptx = compile_to_ptx(program2);
   
    printf("%s\n",ptx);

  
  // get device 0

  err = cuDeviceGet(&device, 0); // or some other device on your system
  if (err != CUDA_SUCCESS) {
        fprintf(stderr, "* Error initializing the CUDA context.\n");
      //  cuCtxDestroy (context);
        exit(-1);
    }

  err = cuCtxCreate(&context, 0, device);
  if (err != CUDA_SUCCESS) {
        fprintf(stderr, "* Error initializing the CUDA context.\n");
        cuCtxDestroy (context);
        exit(-1);
    }

  // The magic happens here:
  
  err = cuModuleLoadDataEx(&module,  ptx, 0, 0, 0);
  if (err != CUDA_SUCCESS) {
        fprintf(stderr, "* Error initializing the CUDA context.\n");
        cuCtxDestroy (context);
        exit(-1);
    }

 

  // And here is how you use your compiled PTX
 
  err = cuModuleGetFunction(&function, module, "map_ske");

  if (err != CUDA_SUCCESS) {
        printf("error: %d\n", err);
        fprintf(stderr, "* Error getting kernel function %s\n", kernel_name);
        cuCtxDestroy (context);
        exit(-1);
    }
  //cuLaunchKernel(kernel_addr, 
   // launch parameters go here
   // kernel arguments go here

   int size = 10;
   int a[size], b[size];
    CUdeviceptr d_a, d_b;

   for (int i = 0; i < size; ++i) {
        a[i] = i;
   }     

   err = cuMemAlloc(&d_a, sizeof(int) * size) ;
    if (err != CUDA_SUCCESS) {
        printf("error: %d\n", err);
        fprintf(stderr, "* Error getting kernel function %s\n", kernel_name);
        cuCtxDestroy (context);
        exit(-1);
    }

   err = cuMemAlloc(&d_b, sizeof(int) * size) ;
    if (err != CUDA_SUCCESS) {
        printf("error: %d\n", err);
        fprintf(stderr, "* Error getting kernel function %s\n", kernel_name);
        cuCtxDestroy (context);
        exit(-1);
    }

   err= cuMemcpyHtoD(d_a, a, sizeof(int) * size) ;
    if (err != CUDA_SUCCESS) {
        printf("error: %d\n", err);
        fprintf(stderr, "* Error getting kernel function %s\n", kernel_name);
        cuCtxDestroy (context);
        exit(-1);
    }

   void *args[3];
   args[0] = (void*)  &d_a;
   args[1] = (void*) &d_b;
   args[2] = (void*) &size;

   err = cuLaunchKernel(function, size, 1, 1,  // Nx1x1 blocks
                                    1, 1, 1,            // 1x1x1 threads
                                    0, 0, args, 0) ;
  
 

   if (err != CUDA_SUCCESS) {
         char message[200];
        const char *error;
        cuGetErrorString(err, &error);
        strcpy(message,"Error create_ref_nif: ");
        strcat(message, error);
        printf("%s\n",error);
        exit(-1);
    }

  cuMemcpyDtoH(b, d_b, sizeof(int) * size) ;

   for (int i = 0; i < size; ++i) {
        printf("result[%d] = %d\n", i, b[i]);
   }     

  cuMemcpyDtoH(a, d_a, sizeof(int) * size) ;

   for (int i = 0; i < size; ++i) {
        printf("result[%d] = %d\n", i, a[i]);
   }      
}  