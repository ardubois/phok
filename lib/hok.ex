defmodule Hok do
  @on_load :load_nifs
  def load_nifs do
      :erlang.load_nif('./priv/gpu_nifs', 0)
      #IO.puts("ok")
  end

  defmacro hok(function) do
    #resp =  Macro.escape(quote(do: {:anon , unquote(function)}))
     #resp
    #IO.inspect function
    #raise "hell"
    {fname,type} = Hok.CudaBackend.gen_lambda("Elixir.App",function)
    result = quote do: Hok.load_lambda_compilation(unquote("Elixir.App"), unquote(fname), unquote(type))
    #IO.inspect result
    #raise "hell"
    result
  end

  defmacro gpufor({:<-, _ ,[var,tensor]},do: b)  do
      quote do: Comp.comp(unquote(tensor), Hok.hok (fn (unquote(var)) -> (unquote b) end))
  end

  defmacro gpufor({:<-,_, [var1, {:..,_, [_b1, e1]}]}, arr1, arr2,do: body) do
       r=      quote do: Comp.comp_xy_2arrays(unquote(arr1), unquote(arr2), unquote(e1),
                                          Hok.hok (fn (unquote(arr1),
                                                       unquote(arr2),
                                                       unquote(var1)) -> (unquote body) end))
      #IO.inspect r
       #raise "hell"
       r
  end

  defmacro gpufor({:<-,_, [var1, {:..,_, [_b1, e1]}]}, {:<-,_, [var2, {:..,_, [_b2, e2]}]},arr1,arr2, par3, do: body) do
       r=      quote do: MM.comp2xy2D1p(unquote(arr1), unquote(arr2), unquote(par3), unquote(e1), unquote(e2),
                                          Hok.hok (fn (unquote(arr1),
                                                       unquote(arr2),
                                                       unquote(par3),
                                                       unquote(var1),
                                                       unquote(var2)) -> (unquote body) end))
       #IO.inspect r
       #raise "hell"
       r
  end


  defmacro defmodule(header,do: body) do
    #IO.inspect header
    #IO.inspect body
    {:__aliases__, _, [module_name]} = header


    code = Hok.CudaBackend.compile_module(module_name,body)

    file = File.open!("c_src/Elixir.#{module_name}.cu", [:write])
    IO.write(file, "#include \"erl_nif.h\"\n\n" <> code)
    File.close(file)
    {result, errcode} = System.cmd("nvcc",
        [ "--shared",
          "--compiler-options",
          "'-fPIC'",
          "-o",
          "priv/Elixir.#{module_name}.so",
          "c_src/Elixir.#{module_name}.cu"
  ], stderr_to_stdout: true)



  if ((errcode == 1) || (errcode ==2)) do raise "Error when compiling .cu file generated by Hok:\n#{result}" end

    ast_new_module = Hok.CudaBackend.gen_new_module(header,body)
    #IO.inspect ast_new_module
    ast_new_module


    #quote do: IO.puts "ok"
  end

  defmacro include(inc_list) do
    #IO.inspect inc_list
    includes = inc_list
                |> Enum.map(fn {_,_,[module]} -> to_string(module) end)

    file = File.open!("c_src/Elixir.App.cu", [:write])
    IO.write(file, "#include \"erl_nif.h\"\n\n")
    Enum.map(includes, fn module ->   code = File.read!("c_src/Elixir.#{module}.cu")
                                          |>  String.split("\n")
                                          |>  Enum.drop(1)
                                          |> Enum.join("\n")
                                      IO.write(file, code)  end)
    File.close(file)

    {result, errcode} = System.cmd("nvcc",
        [ "--shared",
          "--compiler-options",
          "'-fPIC'",
          "-o",
          "priv/Elixir.App.so",
          "c_src/Elixir.App.cu"
  ], stderr_to_stdout: true)


  if ((errcode == 1) || (errcode ==2)) do raise "Error when compiling .cu file generated by Hok:\n#{result}" end


  end

  #####################
  #####
  ##### Legacy code:  gptype macro ##########
  #####
  ##########################

  defmacro deft({func,_,[type]}) do

    if (nil == Process.whereis(:gptype_server)) do
      pid = spawn_link(fn -> gptype_server() end)
      Process.register(pid, :gptype_server)
    end
    send(:gptype_server,{:add_type, func,type_to_list(type)})
    #IO.inspect(type_to_list(type))
    quote do
    end
  end
  def gptype_server(), do: gptype_server_(Map.new())
  defp gptype_server_(map) do
    receive do
      {:add_type, fun, types}  -> map=Map.put(map,fun, types)
                              gptype_server_(map)
      {:get_type, pid,fun} -> type=Map.get(map,fun)
                              send(pid,{:type,fun,type})
                              gptype_server_(map)
      {:kill}               -> :dead
    end
  end
  defp type_to_list({:integer,_,_}), do: [:int]
  defp type_to_list({:unit,_,_}), do: [:unit]
  defp type_to_list({:float,_,_}), do: [:float]
  defp type_to_list({:gmatrex,_,_}), do: [:matrex]
  defp type_to_list([type]), do: [type_to_list(type)]
  defp type_to_list({:~>,_, [a1,a2]}), do: type_to_list(a1) ++ type_to_list(a2)
  defp type_to_list({x,_,_}), do: raise "Unknown type constructor #{x}"
  def is_typed?() do
    nil != Process.whereis(:gptype_server)
  end
  def get_type_kernel(fun_name) do
    send(:gptype_server,{:get_type, self(),fun_name})
    receive do
      {:type,fun,type} -> if fun == fun_name do
                                send(:gptype_server,{:kill})
                                type
                          else
                                raise "Asked for #{fun_name} got #{fun}"
                          end
      end

    end
    def get_type_fun(fun_name) do
      send(:gptype_server,{:get_type, self(),fun_name})
      receive do
        {:type,fun,type} -> if fun == fun_name do
                                  type
                            else
                                  raise "Asked for #{fun_name} got #{fun}"
                            end
        end

      end

  #############################################
  ###########
  #######   Legacy code Hok macro
  #######
  #####################




#defp gen_para(p,:matrex) do
#  "float *#{p}"
#end
#defp gen_para(p,:float) do
#  "float #{p}"
#end
#defp gen_para(p,:int) do
#  "int #{p}"
#end
#defp gen_para(p, list) when is_list(list) do
#  size = length(list)
#
#  {ret,type}=List.pop_at(list,size-1)
#
#  r="#{ret} (*#{p})(#{to_arg_list(type)})"
#  r
#
#end
#defp to_arg_list([t]) do
#  "#{t}"
#end
#defp to_arg_list([v|t]) do
#  "#{v}," <> to_arg_list(t)
#end

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
def new_gnx(size,type) do
  {:nx, type, :shape, :name, :ref}
end
def new_gnx ((%Nx.Tensor{data: data, type: type, shape: shape, names: name}) ) do
 # %Nx.BinaryBackend{ state: array} = data
  #{l,c} = shape
  #ref = case type do
   #  {:f,32} -> create_nx_ref_nif(array,l,c,Kernel.to_charlist("float"))
   #  {:f,64} -> create_nx_ref_nif(array,l,c,Kernel.to_charlist("double"))
   #  {:s,32} -> create_nx_ref_nif(array,l,c,Kernel.to_charlist("int"))
   #  x -> raise "new_gmatrex: type #{x} not suported"
  #end
  {:nx, type, shape, name , :ref}
end

################################
#######
#######  GMatrex stuff:
#######
####################
def new_gpu_nx_nif(_l,_c,_type) do
  raise "NIF get_nx_nif/4 not implemented"
end
def get_nx_nif(_matrex,_l,_c,_type) do
  raise "NIF get_nx_nif/4 not implemented"
end
def create_nx_ref_nif(_matrex,_l,_c,_type) do
  raise "NIF create_nx_ref_nif/4 not implemented"
end
  def create_ref_nif(_matrex) do
    raise "NIF create_ref_nif/1 not implemented"
end
def new_pinned_nif(_list,_length) do
  raise "NIF new_pinned_nif/1 not implemented"
end
def new_gmatrex_pinned_nif(_array) do
  raise "NIF new_gmatrex_pinned_nif/1 not implemented"
end
def new_pinned(list) do
  size = length(list)
  {new_pinned_nif(list,size), {1,size}}
end
def new_gmatrex(%Matrex{data: matrix} = a) do
  ref=create_ref_nif(matrix)
  {ref, Matrex.size(a)}
end
def new_gmatrex((%Nx.Tensor{data: data, type: type, shape: shape, names: name}) ) do
  %Nx.BinaryBackend{ state: array} = data
  {l,c} = shape
  ref = case type do
     {:f,32} -> create_nx_ref_nif(array,l,c,Kernel.to_charlist("float"))
     {:f,64} -> create_nx_ref_nif(array,l,c,Kernel.to_charlist("double"))
     {:s,32} -> create_nx_ref_nif(array,l,c,Kernel.to_charlist("int"))
     x -> raise "new_gmatrex: type #{x} not suported"
  end
  {:nx, type, shape, name , ref}
end
def new_gmatrex({array,{l,c}}) do
  ref=new_gmatrex_pinned_nif(array)
  {ref, {l,c}}
end

def new_gmatrex(r,c) do
  ref=new_ref_nif(c)
  {ref, {r,c}}
  end

def gmatrex_size({_r,{l,size}}), do: {l,size}

def new_ref_nif(_matrex) do
  raise "NIF new_ref_nif/1 not implemented"
end
def synchronize_nif() do
  raise "NIF new_ref_nif/1 not implemented"
end
def synchronize() do
  synchronize_nif()
end
def new_ref(size) do
ref=new_ref_nif(size)
{ref, {1,size}}
end
def get_matrex_nif(_ref,_rows,_cols) do
raise "NIF get_matrex_nif/1 not implemented"
end
def get_gmatrex({ref,{rows,cols}}) do
  %Matrex{data: get_matrex_nif(ref,rows,cols)}
end
def get_gmatrex({:nx, type, shape, name , ref}) do
  {rows,cols} = shape
  %Nx.Tensor{data: %Nx.BinaryBackend{ state: get_matrex_nif(ref,rows,cols)}, type: type, shape: shape, names: name}
end

def load_kernel_nif(_module,_fun) do
  raise "NIF load_kernel_nif/2 not implemented"
end
def load_fun_nif(_module,_fun) do
  raise "NIF load_fun_nif/2 not implemented"
end

############################################################## Loading types and asts from files
def load_ast(kernel) do
  {module,f_name}= case Macro.escape(kernel) do
    {:&, [],[{:/, [], [{{:., [], [module, f_name]}, [no_parens: true], []}, _nargs]}]} -> {module,f_name}
     _ -> raise "Argument to spawn should be a function."
  end
  IO.inspect f_name
  bytes = File.read!("c_src/#{module}.asts")
  map_asts = :erlang.binary_to_term(bytes)

  {ast,_typed?,_types} = Map.get(map_asts,String.to_atom("#{f_name}"))
  ast
end

def load_type_ast(kernel) do
  {:&, _ ,[{:/, _,  [{{:., _, [{:__aliases__, _, [module]}, kernelname]}, _, []}, _nargs]}]} = kernel
  bytes = File.read!("c_src/Elixir.#{module}.types")
  map_types = :erlang.binary_to_term(bytes)

              #module_name=String.slice("#{module}",7..-1//1) # Eliminates Elixir.
  type = Map.get(map_types,String.to_atom("#{kernelname}"))

  bytes = File.read!("c_src/Elixir.#{module}.asts")
  map_asts = :erlang.binary_to_term(bytes)

            #module_name=String.slice("#{module}",7..-1//1) # Eliminates Elixir.
  ast = Map.get(map_asts,String.to_atom("#{kernelname}"))
  {type,ast}
end

def load_type(kernel) do
  case Macro.escape(kernel) do
    {:&, [],[{:/, [], [{{:., [], [module, kernelname]}, [no_parens: true], []}, _nargs]}]} ->
             #IO.inspect module

              bytes = File.read!("c_src/#{module}.types")
              map = :erlang.binary_to_term(bytes)

              #module_name=String.slice("#{module}",7..-1//1) # Eliminates Elixir.


              resp = Map.get(map,String.to_atom("#{kernelname}"))
              #IO.inspect resp
              resp
    _ -> raise "Hok.build: invalid kernel"
  end
end
def load(kernel) do
  case Macro.escape(kernel) do
    {:&, [],[{:/, [], [{{:., [], [_module, kernelname]}, [no_parens: true], []}, _nargs]}]} ->


             # IO.puts module
              #raise "hell"
              #module_name=String.slice("#{module}",7..-1//1) # Eliminates Elixir.
              Hok.load_kernel_nif(to_charlist("Elixir.App"),to_charlist("#{kernelname}"))

    _ -> raise "Hok.build: invalid kernel"
  end
end
def load_fun(fun) do
  case Macro.escape(fun) do
    {:&, [],[{:/, [], [{{:., [], [_module, funname]}, [no_parens: true], []}, _nargs]}]} ->

              #module_name=String.slice("#{module}",7..-1//1) # Eliminates Elixir.

              Hok.load_fun_nif(to_charlist("Elixir.App"),to_charlist("#{funname}"))
    _ -> raise "Hok.invalid function"
  end
end
def load_lambda_compilation(_module,lambda,type) do
 # {:anon, lambda, Hok.load_fun_nif(to_charlist(module),to_charlist(lambda)), type}
 {:anon, lambda, type}
end
def load_lambda(lambda) do
  Hok.load_fun_nif(to_charlist("Elixir.App"),to_charlist(lambda))
 end
############################
######
######   Prepares the  arguments before making the real kernell call
######
##############
defp process_args([{:anon,name,_type}|t1]) do
  [load_lambda(name) | process_args(t1)]
end
defp process_args([{:func, func, _type}|t1]) do
  [load_fun(func)| process_args(t1)]
end
defp process_args([{:nx, _type, _shape, _name , ref}|t1]) do
  [ref| process_args(t1)]
end
defp process_args([{matrex,{_rows,_cols}}| t1]) do
  [matrex | process_args(t1)]
end
defp process_args([arg|t1]) when is_function(arg) do
  [load_fun(arg)| process_args(t1)]
end
defp process_args([arg|t1]) do
  [arg | process_args(t1)]
end
defp process_args([]), do: []

#########################
defp process_args_no_fun([{:anon,_name,_type}|t1]) do
  process_args_no_fun(t1)
end
defp process_args_no_fun([{:func, _func, _type}|t1]) do
  process_args_no_fun(t1)
end
defp process_args_no_fun([ {:nx, _type, _shape, _name , ref}| t1]) do
  [ref | process_args_no_fun(t1)]
end
defp process_args_no_fun([{matrex,{_rows,_cols}}| t1]) do
  [matrex | process_args_no_fun(t1)]
end
defp process_args_no_fun([arg|t1]) when is_function(arg) do
  process_args_no_fun(t1)
end
defp process_args_no_fun([arg|t1]) do
  [arg | process_args_no_fun(t1)]
end
defp process_args_no_fun([]), do: []

############################ Type checking the arguments at runtime

### first two arguments are used for error messages. Takes a list of types and a list of actual parameters and type checks them

def type_check_args(kernel,narg, [:matrex | t1], [a|t2]) do
    case a do
      {_ref,{_l,_c}} -> type_check_args(kernel,narg+1,t1,t2)
      {:nx, _type, _shape, _name , _ref} -> type_check_args(kernel,narg+1,t1,t2)
      _             -> raise "#{kernel}: argument #{narg} should have type gmatrex."
    end

end
def type_check_args(kernel,narg, [:float | t1], [v|t2]) do
    if is_float(v) do
      type_check_args(kernel,narg+1,t1,t2)
    else
      raise "#{kernel}: argument #{narg} should have type float."
    end
end
def type_check_args(kernel,narg, [:int | t1], [v|t2]) do
  if is_integer(v) do
    type_check_args(kernel,narg+1,t1,t2)
  else
    raise "#{kernel}: argument #{narg} should have type int."
  end
end
def type_check_args(kernel,narg, [{rt , ft} | t1], [{:func, func, { art , aft}} |t2]) do
  f_name= case Macro.escape(func) do
    {:&, [],[{:/, [], [{{:., [], [_module, f_name]}, [no_parens: true], []}, _nargs]}]} -> f_name
     _ -> raise "Argument to spawn should be a function."
   end
  if rt == art do

     type_check_function(f_name,0,ft,aft)
     type_check_args(kernel,narg+1,t1,t2)
   else
     raise "#{kernel}: #{f_name} function has return type #{art}, was excpected to have type #{rt}."
   end
  end

def type_check_args(kernel,narg, [{rt , ft} | t1], [{:anon, _name, { art , aft}} |t2]) do
  if rt == art do
    type_check_function("anonymous",1,ft,aft)
    type_check_args(kernel,narg+1,t1,t2)
  else
    raise "#{kernel}: anonymous function has return type #{art}, was excpected to have type #{rt}."
  end
end
def type_check_args(kernel,narg, [{rt , ft} | t1], [func |t2]) when is_function(func) do
  #IO.inspect func
  #raise "hell"
   {art,aft} = load_type(func)
   #IO.inspect ft
   #IO.inspect aft
   f_name= case Macro.escape(func) do
    {:&, [],[{:/, [], [{{:., [], [_module, f_name]}, [no_parens: true], []}, _nargs]}]} -> f_name
     _ -> raise "Argument to spawn should be a function."
   end
  if rt == art do
      type_check_function(f_name,0,ft,aft)
      type_check_args(kernel,narg+1,t1,t2)
    else
      raise "#{kernel}: #{f_name} function has return type #{art}, was excpected to have type #{rt}."
    end
end
def type_check_args(_k,_narg,[],[]), do: []
def type_check_args(k,_narg,a,v), do: raise "Wrong number of arguments when calling #{k}. #{inspect a} #{inspect v} "

def type_check_function(k,narg,[at|t1],[ft|t2]) do
    if (at == ft) do
      type_check_function(k,narg+1,t1,t2)
    else
      raise "#{k}: argument #{narg} has type #{ft} and should have type #{at}"
    end
end
def type_check_function(_k,_narg,[],[]), do: []
def type_check_function(k,_narg,a,v), do: raise "Wrong number of arguments when calling #{k}. #{inspect a} #{inspect v} "

####################### loads type of a function at compilation time
defmacro lt(k) do
  type = load_type_at_compilation(k)
  r= quote do: {:func, unquote(k), unquote(type)}
  #IO.inspect r
  r
end
def load_type_at_compilation(kernel) do
  {:&, _ ,[{:/, _,  [{{:., _, [{:__aliases__, _, [module]}, kernelname]}, _, []}, _nargs]}]} = kernel
#  {:&, [],[{:/, [], [{{:., [], [module, kernelname]}, [no_parens: true], []}, _nargs]}]} = kernel
  bytes = File.read!("c_src/Elixir.#{module}.types")
              map = :erlang.binary_to_term(bytes)

              #module_name=String.slice("#{module}",7..-1//1) # Eliminates Elixir.


              resp = Map.get(map,String.to_atom("#{kernelname}"))
              #IO.inspect resp
              resp
end
####################################


##################################
###########
###########  Spawn with jit compilation
###########
########################################

######## at compilation we build a representation for the kernel: {:ker, its type, its ast}
##### and leave a call to spawn

def spawn_jit(k,t,b,l) do
  ast  = load_ast(k)
  #IO.inspect ast
  IO.inspect l
  delta = JIT.gen_types_delta(ast,l)
  inf_types = JIT.infer_types(ast,delta)
  IO.inspect inf_types
  raise "hell"
  case k do
    {:&, _,_} ->
            #IO.inspect t
            {type,{ast,typed?,delta}} = load_type_ast(k)
            #IO.inspect type
            n_ast = Macro.escape ast
            n_delta = Macro.escape delta
            result =  quote do: Hok.spawn({:ker,unquote(k),(unquote type),
                                      {(unquote n_ast), (unquote typed?), unquote(n_delta)}},
                                      unquote(t),unquote(b), unquote(l))
            result
    _ -> IO.inspect k
       raise "The first argumento to spawn should be a Hok kernel: &Module.kernel/nargs"
  end
end



#############################################3
##########
########### spawn that uses function pointers
#########
#######################################

def spawn({:func, k, type}, t,b,l) do
  IO.puts "Aqui!"
  f_name= case Macro.escape(k) do
    {:&, [],[{:/, [], [{{:., [], [_module, f_name]}, [no_parens: true], []}, _nargs]}]} -> f_name
     _ -> raise "Argument to spawn should be a function."
  end



    {:unit,tk} = type

    type_check_args(f_name,1,tk,l)



    args = process_args(l)
    pk=load(k)
    spawn_nif(pk,t,b,args)

end

def spawn(k,t,b,l) when is_function(k) do
   #IO.inspect k
   #raise "hell"

  f_name= case Macro.escape(k) do
    {:&, [],[{:/, [], [{{:., [], [_module, f_name]}, [no_parens: true], []}, _nargs]}]} -> f_name
     _ -> raise "Argument to spawn should be a function."
  end



    {:unit,tk} = load_type(k)

    type_check_args(f_name,1,tk,l)

    pk=load(k)

    args = process_args(l)

    spawn_nif(pk,t,b,args)

end
def spawn(_k,_t,_b,_l) do
  #IO.inspect _k
  raise "First argument of spawn must be a function.."
end
def spawn_nif(_k,_t,_b,_l) do
  raise "NIF spawn_nif/1 not implemented"
end


end
