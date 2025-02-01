require Hok

Hok.defmodule_jit PMap do
  defk map_ske(a1,a2,size,f) do
    var id int = blockIdx.x * blockDim.x + threadIdx.x

    if(id < size) do
      a2[id] = f(a1[id])
    end
  end
  def map(v1, f) do
    {l,c} = Hok.get_shape_gnx(v1)
    type = Hok.get_type_gnx(v1)
    #IO.inspect shape
    #raise "hell"
    size = l*c

    threadsPerBlock = 128;
    numberOfBlocks = div(size + threadsPerBlock - 1, threadsPerBlock)

    result_gpu =Hok.new_gnx(l,c,type)



    Hok.spawn_jit(&PMap.map_ske/4,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[v1,result_gpu,size, f])
    result_gpu

  end

end

#a = Hok.hok (fn x,y -> x+y end)
#IO.inspect a
#raise "hell"

tensor1 = Nx.tensor([[1,2,3,4]],type: {:s, 32})
tensor2 = Nx.tensor([[1,2,3,4]],type: {:f, 32})
tensor3 = Nx.tensor([[1,2,3,4]],type: {:f, 64})
gtensor1 = Hok.new_gnx(tensor1)
gtensor2 = Hok.new_gnx(tensor2)
gtensor3 = Hok.new_gnx(tensor3)

func = Hok.hok fn (x) -> x + 1 end

#PMap.map(gtensor,&PMap.inc/1)

prev = System.monotonic_time()

gtensor1
    |> PMap.map(func)
    |> Hok.get_gnx
    |> IO.inspect

gtensor2
    |> PMap.map(func)
    |> Hok.get_gnx
    |> IO.inspect

gtensor3
    |> PMap.map(func)
    |> Hok.get_gnx
    |> IO.inspect
PMap.map(gtensor1,func)
r2 = PMap.map(gtensor2,func)
r3 = PMap.map(gtensor3,func)
#r = Hok.new_gnx(1,4,{:s,32})


IO.inspect r
next = System.monotonic_time()
IO.puts "Hok\t\t#{System.convert_time_unit(next-prev,:native,:millisecond)}"
