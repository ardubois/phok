defmodule JIT do

def infer_types({:defk,_,[header,[body]]},delta) do
  Hok.TypeInference.type_check(delta,body)
end
  # finds the types of the actual parameters and generates a maping of formal parameters to their types
def gen_types_delta({:defk,_,[header,[body]]}, actual_param) do
  {_, _, formal_para} = header
  types = infer_types_actual_parameters(actual_param)
  formal_para
          |> Enum.map(fn({p, _, _}) -> p end)
          |> Enum.zip(types)
          |> Map.new()
end


def infer_types_actual_parameters([])do
  []
end
def infer_types_actual_parameters([h|t])do
  case h do
    {:nx, type, shape, name , :ref} ->
        case type do
          {:f,32} -> [:tfloat | infer_types_actual_parameters(t)]
          {:f,64} -> [:tdouble | infer_types_actual_parameters(t)]
          {:s,32} -> [:tint | infer_types_actual_parameters(t)]
        end
    float when  is_float(float) ->
        [:float | infer_types_actual_parameters(t)]
    int   when  is_integer(int) ->
        [:int | infer_types_actual_parameters(t)]
    func when is_function(func) ->
        [:none | infer_types_actual_parameters(t)]
  end
end
#########################3 OLD

def compile_and_load_kernel({:ker, _k, k_type,{ast, is_typed?, delta}},  l) do

 # get the formal parameters of the function

  formal_par = get_args(ast)

 # get types of parameters:

  {:unit, type} = k_type


  # creates a map with the names that must be substituted (all parameters that are functions)


  map = create_map_subs(type, formal_par, l, %{})

 #  removes the arguments that will be substituted from the kernel definition

 ast = remove_args(map,ast)

 # makes the substitutions:

  ast = subs(map, ast)


  r = gen_jit_kernel_load(ast, is_typed?, delta)
  r
end

def gen_jit_kernel_load({:defk,_,[header,[body]]}, is_typed, inf_types) do

  {kname, _, para} = header

  param_list = para
       |> Enum.map(fn {p, _, _}-> Hok.CudaBackend.gen_para(p,Map.get(inf_types,p)) end)
       |> Enum.join(", ")

  types_para = para
       |>  Enum.map(fn {p, _, _}-> Map.get(inf_types,p) end)


  fname = "ker_" <> Hok.CudaBackend.gen_lambda_name()
  #fname = "k072b2a4iad"
  #fname = Hok.CudaBackend.gen_lambda_name()
  cuda_body = Hok.CudaBackend.gen_cuda(body,inf_types,is_typed,"")
  k = Hok.CudaBackend.gen_kernel(fname,param_list,cuda_body)
  accessfunc = Hok.CudaBackend.gen_kernel_call(fname,length(para),Enum.reverse(types_para))
  code = "\n" <> k <> "\n\n" <> accessfunc

 # IO.puts code
  file = File.open!("c_src/Elixir.App.cu", [:append])
  IO.write(file, "//#############################\n\n" <> code)
  File.close(file)
  {result, errcode} = System.cmd("nvcc",
        [ "--shared",
          "--compiler-options",
          "'-fPIC'",
          "-o",
          "priv/Elixir.App.so",
          "c_src/Elixir.App.cu"
    ], stderr_to_stdout: true)


    if ((errcode == 1) || (errcode ==2)) do raise "Error when JIT compiling .cu file generated by Hok: #{kname}\n #{result}" end
    IO.puts "antes"
    r = Hok.load_kernel_nif(to_charlist("Elixir.App"),to_charlist("#{fname}"))
    IO.puts "depois"
    #Hok.load_kernel_nif(to_charlist("Elixir.App"),to_charlist("map_kernel"))
    #Hok.load_fun_nif(to_charlist("Elixir.App"),to_charlist("#{fname}_call"))
    r
end
############## Removing from kernel definition the arguments that are functions
def remove_args(map, ast) do
   case ast do
        {:defk, info,[ {name, i2,  args} ,block]} ->  {:defk, info,[ {name, i2, filter_args(map,args)} ,block]}
        _ -> raise "Recompiling kernel: unknown ast!"
   end

end

def filter_args(map,[{var,i, nil}| t]) do
  if map[var] ==  nil do
    [{var,i, nil}| filter_args(map,t)]
  else
    filter_args(map,t)
  end
end
def filter_args(_map,[]), do: []

def get_args(ast) do
  case ast do
       {:defk, _info,[ {_name, _i2,  args} ,_block]} ->  args
       _ -> raise "Recompiling kernel: unknown ast!"
  end

end

#######################
#########
######### Creates a map with the substitutions to be made: formal parameter => actual paramenter
########
#######################
def create_map_subs([{_rt, funct} |tt], [{fname,_,nil} | tfa], [{:func, func, _type} | taa], map) when is_list(funct) and is_function(func) do
  case Macro.escape(func) do
    {:&, [],[{:/, [], [{{:., [], [_module, func_name]}, [no_parens: true], []}, _nargs]}]} ->
        create_map_subs(tt,tfa,taa,Map.put(map,fname,func_name))
    _ -> raise "Problem with paramenter #{inspect func}"

  end
end
def create_map_subs([_funct |tt], [{fname,_,nil} | tfa], [func | taa], map) when   is_function(func) do
  case Macro.escape(func) do
    {:&, [],[{:/, [], [{{:., [], [_module, func_name]}, [no_parens: true], []}, _nargs]}]} ->
        create_map_subs(tt,tfa,taa,Map.put(map,fname,func_name))
    _ -> raise "Problem with paramenter #{inspect func}"

  end
end
def create_map_subs([_funct |tt], [{fname,_,nil} | tfa], [{:anon, lambda,_type} | taa], map)  do
         # IO.inspect "yoooooo"
          #raise "hell"
          create_map_subs(tt,tfa,taa,Map.put(map,fname,lambda))
end
def create_map_subs([_t |tt], [_fa | tfa], [_aa | taa], map)  do
    create_map_subs(tt,tfa,taa,map)
end
def create_map_subs([], [], [], map), do: map
def create_map_subs(_,_,_,_), do: raise "spawn: wrong number of parameters at kernel launch."

###################
################### substitute variables that represent functions by the actual function names
############   (substitutes formal parameters that are functions by their actual values)
#### Takes the map created with create_map_subs and the ast and returns a new ast
########################

def subs(map,{:defk, i1,[header, [body]]}) do
   nbody = subs_body(map,body)
   {:defk, i1,[header, [nbody]]}
end


def subs_body(map,body) do


  case body do

      {:__block__, _, _code} ->
        subs_block(map,body)
      {:do, {:__block__,pos, code}} ->
        {:do, subs_block(map, {:__block__, pos,code}) }
      {:do, exp} ->
        {:do, subs_command(map,exp)}
      {_,_,_} ->
        subs_command(map,body)
   end


end
defp subs_block(map,{:__block__, info, code}) do
  {:__block__, info,
      Enum.map(code,  fn com -> subs_command(map,com) end)
  }
end

defp subs_command(map,code) do
    case code do
        {:for,i,[param,[body]]} ->
          {:for,i,[param,[subs_body(map,body)]]}
        {:do_while, i, [[doblock]]} ->
          {:do_while, i, [[subs_body(map,doblock)]]}
        {:do_while_test, i, [exp]} ->
          {:do_while_test, i, [subs_exp(map,exp)]}
        {:while, i, [bexp,[body]]} ->
          {:while, i, [subs_exp(map,bexp),[subs_body(map,body)]]}
        # CRIAÇÃO DE NOVOS VETORES
        {{:., i1, [Access, :get]}, i2, [arg1,arg2]} ->
          {{:., i1, [Access, :get]}, i2, [subs_exp(map,arg1),subs_exp(map,arg2)]}
        {:__shared__, i1, [{{:., i2, [Access, :get]}, i3, [arg1,arg2]}]} ->
          {:__shared__,i1 , [{{:., i2, [Access, :get]}, i3, [subs_exp(map,arg1),subs_exp(map,arg2)]}]}

        # assignment
        {:=, i1, [{{:., i2, [Access, :get]}, i3, [{array,a1,a2},acc_exp]}, exp]} ->
          {:=, i1, [{{:., i2, [Access, :get]}, i3, [{array,a1,a2},subs_exp(map,acc_exp)]}, subs_exp(map,exp)]}
        {:=, i, [var, exp]} ->
          {:=, i, [var, subs_exp(map,exp)]}
        {:if, i, if_com} ->
          {:if, i, subs_if(map,if_com)}
        {:var, i1 , [{var,i2,[{:=, i3, [{type,ii,nil}, exp]}]}]} ->
          {:var, i1 , [{var,i2,[{:=, i3, [{type,ii,nil}, subs_exp(map,exp)]}]}]}
        {:var, i1 , [{var,i2,[{:=, i3, [type, exp]}]}]} ->
          {:var, i1 , [{var,i2,[{:=, i3, [type, subs_exp(map,exp)]}]}]}
        {:var, i1 , [{var,i2,[{type,i3,t}]}]} ->
          {:var, i1 , [{var,i2,[{type,i3,t}]}]}
        {:var, i1 , [{var,i2,[type]}]} ->
          {:var, i1 , [{var,i2,[type]}]}
        {:type, i1 , [{var,i2,[{type,i3,t}]}]} ->
          {:type, i1 , [{var,i2,[{type,i3,t}]}]}
        {:type, i1 , [{var,i2,[type]}]} ->
          {:type, i1 , [{var,i2,[type]}]}

        {:return,i,[arg]} ->
          {:return,i,[subs_exp(map,arg)]}

        {fun, info, args} when is_list(args)->
          new_name = map[fun]
          if (new_name == nil ) do
            {fun, info, Enum.map(args,fn(exp) -> subs_exp(map,exp) end)}
          else
            {new_name, info, Enum.map(args,fn(exp) -> subs_exp(map,exp) end)}
          end
        number when is_integer(number) or is_float(number) -> raise "Error: number is a command"
        {str,i1 ,a } -> {str,i1 ,a }

    end
end

defp subs_if(map,[bexp, [do: then]]) do
  [subs_exp(map,bexp), [do: subs_body(map,then)]]
end
defp subs_if(map,[bexp, [do: thenbranch, else: elsebranch]]) do
  [subs_exp(map,bexp), [do: subs_body(map,thenbranch), else: subs_body(map,elsebranch)]]
end


defp subs_exp(map,exp) do
    case exp do
      {{:., i1, [Access, :get]}, i2, [arg1,arg2]} ->
          {{:., i1, [Access, :get]}, i2, [arg1, subs_exp(map,arg2)]}
      {{:., i1, [{struct, i2, nil}, field]},i3,[]} ->
          {{:., i1, [{struct, i2, nil}, field]},i3,[]}
      {{:., i1, [{:__aliases__, i2, [struct]}, field]}, i3, []} ->
        {{:., i1, [{:__aliases__, i2, [struct]}, field]}, i3, []}
      {op,info, args} when op in [:+, :-, :/, :*] ->
        {op,info, Enum.map(args, fn e -> subs_exp(map,e) end)}
      {op, info, args} when op in [ :<=, :<, :>, :>=, :&&, :||, :!,:!=,:==] ->
        {op,info, Enum.map(args, fn e -> subs_exp(map,e) end)}
      {var,info, nil} when is_atom(var) ->
        {var, info, nil}
      {fun,info, args} ->
        new_name = map[fun]
        if (new_name == nil ) do
          {fun, info, Enum.map(args,fn(exp) -> subs_exp(map,exp) end)}
        else
          {new_name, info, Enum.map(args,fn(exp) -> subs_exp(map,exp) end)}
        end
      float when  is_float(float) -> float
      int   when  is_integer(int) -> int
      string when is_binary(string)  -> string
    end

  end


end
