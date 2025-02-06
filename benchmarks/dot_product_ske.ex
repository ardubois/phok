require Hok
#Nx.default_backend(EXLA.Backend)
#import Nx
Hok.defmodule_jit DP do
include CAS
  defk map_2kernel(a1,a2,a3,size,f) do
    id = blockIdx.x * blockDim.x + threadIdx.x
    if(id < size) do
      a3[id] = f(a1[id],a2[id])
    end
  end
  def map2(t1,t2,func) do

    {l,c} = Hok.get_shape_gnx(t1)
    type = Hok.get_type_gnx(t2)
     size = l*c
     result_gpu = Hok.new_gnx(l,c, type)

      threadsPerBlock = 256;
      numberOfBlocks = div(size + threadsPerBlock - 1, threadsPerBlock)

      Hok.spawn_jit(&DP.map_2kernel/5,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[t1,t2,result_gpu,size,func])


      result_gpu
  end
  def reduce(ref,  f) do

     {l,c} = Hok.get_shape_gnx(ref)
     type = Hok.get_type_gnx(ref)
     size = l*c
      result_gpu  = Hok.new_gnx(Nx.tensor([[0]] , type: type))

      threadsPerBlock = 256
      blocksPerGrid = div(size + threadsPerBlock - 1, threadsPerBlock)
      numberOfBlocks = blocksPerGrid
      Hok.spawn_jit(&DP.reduce_kernel/4,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[ref, result_gpu, f, size])
      result_gpu
  end
  defk reduce_kernel(a, ref4, f,n) do

    __shared__ cache[256]

    tid = threadIdx.x + blockIdx.x * blockDim.x;
    cacheIndex = threadIdx.x

    temp =0.0

    while (tid < n) do
      temp = f(a[tid], temp)
      tid = blockDim.x * gridDim.x + tid
    end

    cache[cacheIndex] = temp
      __syncthreads()

    i = blockDim.x/2

    while (i != 0 ) do  ###&& tid < n) do
      #tid = blockDim.x * gridDim.x + tid
      if (cacheIndex < i) do
        cache[cacheIndex] = f(cache[cacheIndex + i] , cache[cacheIndex])
      end

    __syncthreads()
    i = i/2
    end

  if (cacheIndex == 0) do
    current_value = ref4[0]
    while(!(current_value == atomic_cas(ref4,current_value,f(cache[0],current_value)))) do
      current_value = ref4[0]
    end
  end

  end
  def replicate(n, x), do: (for _ <- 1..n, do: x)
end

#Hok.include [DP]

[arg] = System.argv()

n = String.to_integer(arg)


{vet1,_} = Nx.Random.uniform(Nx.Random.key(1), shape: {1, n}, type: :f32)
{vet2,_} = Nx.Random.uniform(Nx.Random.key(1), shape: {1, n}, type: :f32)

#vet1 = Nx.iota({1,n}, type: :f32)
#vet2 = Nx.iota({1,n}, type: :f32)

#vet1 = Hok.new_nx_from_function(1,n,{:f,32},fn -> :rand.uniform(1000) end )
#vet2 = Hok.new_nx_from_function(1,n,{:f,32},fn -> :rand.uniform(1000) end)

#vet1 = Hok.new_nx_from_function(1,n,{:f,32},fn -> 1.0 end )
#vet2 = Nx.tensor([Enum.to_list(1..n)], type: {:f,32})

prev = System.monotonic_time()


ref1 = Hok.new_gnx(vet1)

ref2 = Hok.new_gnx(vet2)


result = ref1
    |> DP.map2(ref2, Hok.hok fn (a,b) -> a * b end)
    |> DP.reduce(Hok.hok fn (a,b) -> a + b end)
    |> Hok.get_gnx

IO.inspect result

next = System.monotonic_time()


IO.puts "Hok\t#{n}\t#{System.convert_time_unit(next-prev,:native,:millisecond)}"
