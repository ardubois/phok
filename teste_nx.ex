require Hok

Nx.tensor([[1,2,3,4]],type: {:s, 32})
  |> PHok.new_gnx
  |> PHok.get_gnx
  |> IO.inspect

Nx.tensor([[1,2,3,4]],type: {:f, 32})
  |> PHok.new_gnx
  |> PHok.get_gnx
  |> IO.inspect


Nx.tensor([[1,2,3,4]],type: {:f, 64})
  |> PHok.new_gnx
  |> PHok.get_gnx
  |> IO.inspect
