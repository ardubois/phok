require Hok


tensor = Nx.tensor([[1,2,3,4]],type: {:s, 32})

gtensor = Hok.new_gnx_fake(tensor)

r = Hok.get_gnx(r)
IO.inspect r
