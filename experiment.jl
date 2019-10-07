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


function build_experiment(actions::Array{Function, 1}, step_size=1)

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

