require Hok

Nx.tensor([[1,2,3,4]],type: {:s, 32})
  |> Hok.new_gnx
  |> Hok.get_gnx
  |> IO.inspect
