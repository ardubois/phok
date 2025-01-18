require Hok
Hok.defmodule_jit PMap do
  defh g(a) do
    a+a
  end
  defh inc(a)do
    v = g(a+a)
    return 1+a
  end
  deft map_ske tfloat ~> tfloat ~> integer ~> [float ~> float] ~> unit
  defk map_ske(a1,a2,size,f) do
    var id int = blockIdx.x * blockDim.x + threadIdx.x
    #var r float = g(a1[id])
    if(id < size) do
      a2[id] = f(a1[id])
    end
  end
  def map(v1, f) do
    shape = Hok.get_shape_gnx(v1)
    type = Hok.get_type_gnx(v1)
   # IO.inspect shape
    #raise "hell"
    {size} = shape
    threadsPerBlock = 128;
    numberOfBlocks = div(size + threadsPerBlock - 1, threadsPerBlock)

    result_gpu =Hok.new_gnx(size,type)

    Hok.spawn_jit(&PMap.map_ske/4,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[v1,result_gpu,size, f])
    #result_gpu
  end

end

tensor = Nx.tensor([1,2,3,4],type: {:s, 32})

gtensor = Hok.new_gnx(tensor)

PMap.map(gtensor,&PMap.inc/1)
