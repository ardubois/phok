defmodule PHok do
  @on_load :load_nifs
  def load_nifs do
      :erlang.load_nif('./priv/phok_nifs', 0)
  end

######################################
#########
########   GNX stuff
#######
##############################

def get_type_gnx({:nx, type, _shape, _name , _ref}) do
  type
end
def get_shape_gnx({:nx, _type, shape, _name , _ref}) do
  shape
end
def new_gnx((%Nx.Tensor{data: data, type: type, shape: shape, names: name}) ) do
  %Nx.BinaryBackend{ state: array} = data
 # IO.inspect name
 # raise "hell"
  {l,c} = case shape do
    {c} -> {1,c}
    {l,c} -> {l,c}
  end
  ref = case type do
     {:f,32} -> create_gpu_array_nx_nif(array,l,c,Kernel.to_charlist("float"))
     {:f,64} -> create_gpu_array_nx_nif(array,l,c,Kernel.to_charlist("double"))
     {:s,32} -> create_gpu_array_nx_nif(array,l,c,Kernel.to_charlist("int"))
     x -> raise "new_gmatrex: type #{x} not suported"
  end
  {:nx, type, shape, name , ref}
end
def new_gnx(l,c,type) do

  ref = case type do
    {:f,32} -> new_gpu_array_nif(l,c,Kernel.to_charlist("float"))
    {:f,64} -> new_gpu_array_nif(l,c,Kernel.to_charlist("double"))
    {:s,32} -> new_gpu_array_nif(l,c,Kernel.to_charlist("int"))
    x -> raise "new_gmatrex: type #{x} not suported"
 end

 {:nx, type, {l,c}, [nil] , ref}
end

def get_gnx({:nx, type, shape, name , ref}) do
  {l,c} = shape
  ref = case type do
    {:f,32} -> get_gpu_array_nif(ref,l,c,Kernel.to_charlist("float"))
    {:f,64} -> get_gpu_array_nif(ref,l,c,Kernel.to_charlist("double"))
    {:s,32} -> get_gpu_array_nif(ref,l,c,Kernel.to_charlist("int"))
    x -> raise "new_gnx: type #{x} not suported"
 end

  %Nx.Tensor{data: %Nx.BinaryBackend{ state: ref}, type: type, shape: shape, names: name}
end

def new_gnx_fake(_size,type) do
  {:nx, type, :shape, :name, :ref}
end
def new_gnx_fake ((%Nx.Tensor{data: _data, type: type, shape: shape, names: name}) ) do
 # %Nx.BinaryBackend{ state: array} = data
  #{l,c} = shape
  #ref = case type do
   #  {:f,32} -> create_gpu_array_nx_nif(array,l,c,Kernel.to_charlist("float"))
   #  {:f,64} -> create_gpu_array_nx_nif(array,l,c,Kernel.to_charlist("double"))
   #  {:s,32} -> create_gpu_array_nx_nif(array,l,c,Kernel.to_charlist("int"))
   #  x -> raise "new_gmatrex: type #{x} not suported"
  #end
  {:nx, type, shape, name , :ref}
end

def new_gpu_array_nif(_l,_c,_type) do
  raise "NIF get_gpu_array_nif/4 not implemented"
end
def get_gpu_array_nif(_matrex,_l,_c,_type) do
  raise "NIF get_gpu_array_nif/4 not implemented"
end
def create_gpu_array_nx_nif(_matrex,_l,_c,_type) do
  raise "NIF create_gpu_array_nx_nif/4 not implemented"
end
  def create_ref_nif(_matrex) do
    raise "NIF create_ref_nif/1 not implemented"
end

end
