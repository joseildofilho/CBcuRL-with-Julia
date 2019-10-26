include("grid_world.jl")

using ModelingToolkit
using Latexify

function inject_0!(p)
	p[end] = 0.0
	p
end

function inject_value!(p)
	p[end] = 0.333
	p
end

actions() = [
			 p -> inject_0!(p),
			 p -> inject_value!(p)
			 ]

function reward(state::Tuple)
	if (state[1] == 1 || state[2] == 1) || (state[1] > 9 || state[2] > 9)
		-1
	else
		1
	end
end

function build(args::Dict)
	species::Integer = args["species"]
	concentrations::Integer = args["concentrations"]

	@parameters t Cin[1:concentrations]
	@variables N[1:species](t) C[1:concentrations](t)
	@derivatives D'~t

	q = args["q"]

	γ0 = args["γ0"]
	γ = args["γ"]

	K = args["K"]

	μmax = args["μmax"]

	μ = [
		 (μmax[i] * (*)([C[j] / (C[j] + K[i][j]) for j in 1:length(K) if K[i][j] != 0.0]...)) for i in 1:species
		 ]

	Ds = [
		 D(N[i]) ~ N[i] * (μ[i] - q) for i in 1:species
		  ]
	Cs = [
		 D(C[1]) ~ q*(Cin[1] - C[1]) - (+)([((μ[i] * N[i]) / (γ0[i])) 
										   for i in 1:species]...)
		  ]
	c = 2
	for (specie, γ_) in enumerate(γ)
		if !(0.0 == γ_)
			push!(Cs, D(C[c]) ~ q*(Cin[c] - C[c]) - (μ[specie] * N[specie]) / (γ_))
			c += 1
		end
	end

	eqs = [
		   Ds...,
		   Cs...
		   ]

	de = ODESystem(eqs)

	f  = ODEFunction(de, [N..., C...], [Cin...])

	u0    = [100000., 1000., 0.0, 0.0]
	p     = [0.25, 
			 0.]

	(f, u0, p, de)
end
