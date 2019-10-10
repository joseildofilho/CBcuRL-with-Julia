include("simple_auxotroph.jl")
using Plots

mutable struct Experiment
	U
	T
	Experiment() = new([],[])
	Experiment(x,y) = new(x,y)
end

@recipe f(exp::Experiment) = (vcat(exp.T...), transpose(hcat(exp.U...)))

function (e::Experiment)(sol)
	append!(e.T, sol.t)
	append!(e.U, sol.u)
end

function get_state(states::Array; factor::Real=1e6, n::Integer=10)

	if length(states) == 0
		throw(ArgumentError("This Array must have at least 1 item"))
	elseif n <= 0
		throw(ArgumentError("This n must be bigger than 0 item"))
	elseif length(filter(x -> x < 0, states)) != 0
		throw(ArgumentError("This Array cannot have negative values"))
	elseif abs(factor) ≈ 0
		throw(ArgumentError("The factor must be different from 0"))
	end

	x = []
	for state in states
		aux = convert(Integer, round(state / factor))
		if aux > n
			aux = n
		end
		append!(x, aux)
	end
	Tuple(i for i in x)
end

function build_experiment(actions::Array{Function, 1}, step_size::Integer=1)

	f, u0, p = build()

	u::Array = copy(u0)

	exp::Experiment = Experiment()
	step::Float64 	= 0
	tspan::Tuple    = (0., step_size)

	function step!(action::Int)

		actions[action](p)

		prob  = ODEProblem(f,u, tspan, p)
		sol = solve(prob)
		exp(sol)

		step += step_size

		tspan = (step, step + step_size)
		u 	  = exp.U[end]
		u[u .< 0] .= 0
		u[isapprox.(u, 0, atol=1e-2)] .= 0
		exp
	end

	function reset!() 
		u     = copy(u0)
		exp   = Experiment()
		step  = 0
		tspan = (0., step_size)
	end
	get_state(exp::Experiment, populations = 2, unit = 10^6) =
			div.(exp.U[end][1:populations], unit)

	exp, step!, reset!
end

