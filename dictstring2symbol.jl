dictstring2symbol(dict::Dict) = Dict(Symbol(key) => value for (key, value) in dict)
