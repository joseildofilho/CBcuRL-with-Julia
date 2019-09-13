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

function build_experiment(step_size=5)

	f, u0, tspan, p = build()
	exp::Experiment = Experiment()
	step::Float64 	= 0

	function step!()
		prob  = ODEProblem(f,u0, tspan, p)

		sol = solve(prob)
		exp(sol)

		step += step_size

		tspan = (step, step + step_size)
		u0 	  = exp.U[end]
		exp
	end

	exp, step!
end

get_state(exp::Experiment, populations::Int32 = 2, unit::Int32 = 1e6) = div.(exp.U[end][1:populations], unit)
