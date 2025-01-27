require Hok
Hok.defmodule_jit PMap do
  defh cc(a) do
    a+a
  end
  defh g(a) do
    cc(a+a)
  end
  defh inc(a)do
    #v = g(a+a)
    1+a
  end
  deft map_ske tfloat ~> tfloat ~> integer ~> [float ~> float] ~> unit
  defk map_ske(a1,a2,size,f) do
    var id int = blockIdx.x * blockDim.x + threadIdx.x
    var r int = g(a1[id])
    if(id < size) do
      a2[id] = f(a1[id])
    end
  end
  def map(v1, f) do
    shape = Hok.get_shape_gnx(v1)
    type = Hok.get_type_gnx(v1)
    IO.inspect shape
    #raise "hell"
    {size} = shape

    threadsPerBlock = 128;
    numberOfBlocks = div(size + threadsPerBlock - 1, threadsPerBlock)

    result_gpu =Hok.new_gnx(1,size,type)



    Hok.spawn_jit(&PMap.map_ske/4,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[v1,result_gpu,size, f])
    #result_gpu
  end

end

#a = Hok.hok (fn x,y -> x+y end)
#IO.inspect a
#raise "hell"

tensor = Nx.tensor([1,2,3,4],type: {:s, 32})

gtensor = Hok.new_gnx(tensor)

func = Hok.hok fn (x) -> x + 1 end

#PMap.map(gtensor,&PMap.inc/1)

prev = System.monotonic_time()
r = PMap.map(gtensor,func)
IO.inspect r
next = System.monotonic_time()
IO.puts "Hok\t\t#{System.convert_time_unit(next-prev,:native,:millisecond)}"
