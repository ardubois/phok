require Hok
defmodule BMP do
  @on_load :load_nifs
  def load_nifs do
      :erlang.load_nif('./priv/bmp_nifs', 0)
  end
  def gen_bmp_nif(_string,_dim,_mat) do
      raise "gen_bmp_nif not implemented"
  end
  def gen_bmp(string,dim,%Matrex{data: matrix} = _a) do
    gen_bmp_nif(string,dim,matrix)
  end
end

Hok.defmodule_jit Julia do
  defh julia(x,y,dim) do
    scale  = 0.1
    jx = scale * (dim - x)/dim
    jy = scale * (dim - y)/dim

    cr  = -0.8
    ci  = 0.156
    ar  = jx
    ai  = jy
    for i in range(0,200) do
        nar = (ar*ar - ai*ai) + cr
        nai = (ai*ar + ar*ai) + ci
        if ((nar * nar)+(nai * nai ) > 1000.0) do
          return 0
        end
        ar = nar
        ai = nai
    end
    1
  end
  defh julia_function(ptr,x,y,dim) do
    offset = x + y * dim # gridDim.x
    juliaValue = julia(x,y,dim)

    ptr[offset*4 + 0] = 255 * juliaValue;
    ptr[offset*4 + 1] = 0;
    ptr[offset*4 + 2] = 0;
    ptr[offset*4 + 3] = 255;
    return 1
  end


  defk mapgen2D_xy_1para_noret_ker(resp,arg1,size,f)do
    var x int= blockIdx.x * blockDim.x + threadIdx.x
    var y int  = blockIdx.y * blockDim.y + threadIdx.y

    if(x < size && y < size) do
      var v int=f(resp,x,y,arg1)
    end
  end
  def mapgen2D_step_xy_1para_noret(result_gpu,step, arg1, size,f) do



    Hok.spawn(&Julia.mapgen2D_xy_1para_noret_ker/4,{size,size,1},{1,1,1},[result_gpu,arg1,size,f])
    result_gpu
  end
end

Hok.include [Julia]

[arg] = System.argv()
m = String.to_integer(arg)

dim = m

values_per_pixel = 4

result_gpu = Hok.new_gnx(size*size,4,{:s,32})

prev = System.monotonic_time()

d_image = Julia.mapgen2D_step_xy_1para_noret(4,dim,dim, &Julia.julia_function/4)

image = Hok.get_gnx(d_image)

#image = Hok.get_gmatrex(ref)
next = System.monotonic_time()

IO.puts "Hok\t#{dim}\t#{System.convert_time_unit(next-prev,:native,:millisecond)}"

#BMP.gen_bmp('julia2gpotion.bmp',dim,image)
