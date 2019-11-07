using Plots, LSODA

mutable struct Experiment
	N::Array
	C::Array
	T::Array
	n::Integer
	Experiment() = new([],[],[], 1)
	Experiment(x,y,z) = new(x,y,z)
end

@recipe f(exp::Experiment) = (vcat(exp.T...), transpose(hcat(exp.N...)))

function (e::Experiment)(sol)
	append!(e.T, sol.t)
	append!(e.N, map(sol.u) do a a[1:e.n] end)
	append!(e.C, map(sol.u) do a a[e.n+1:end] end)
end

function get_u(exp::Experiment)
	vcat(exp.N[end], exp.C[end])
end

function (e1::Experiment)(e2::Experiment)
	e1.T = e2.T
	e1.C = e2.C
	e1.N = e2.N
end

function get_state(states::Array; factor::Real=1, n::Integer=10)
	if length(states) == 0
		throw(ArgumentError("This Array must have at least 1 item"))
	elseif n <= 0
		throw(ArgumentError("This n must be bigger than 0 item"))
	elseif length(filter(x -> x < 0, states)) != 0
		@show states
		throw(ArgumentError("This Array cannot have negative values"))
	elseif abs(factor) ≈ 0
		throw(ArgumentError("The factor must be different from 0"))
	end

	x = []
	for state in states
		aux = convert(Integer, ceil(state / factor))
		if aux > n
			aux = n
		end
		append!(x, aux)
	end
	Tuple(i for i in x)
end

function build_experiment(actions::Array{Function, 1}, 
						  reward::Function,
						  envoriment::Dict,
						  u0::Array,
						  p::Array;
						  step_size::Integer=3,
						  episode_size::Integer=1_000,
						  number_species::Integer=2)

	f = envoriment["f"]

	u::Array = copy(u0)

	exp::Experiment = Experiment()
	exp.n = number_species
	step::Float64 	= 0
	tspan::Tuple    = (0., step_size)

	steps::Int64 = 0

	function step!(action::Integer)

		actions[action](p)

		prob  = ODEProblem(f,u, tspan, p)
		sol = solve(prob,lsoda(), abstol=1e-13, reltol=1e-13)
		exp(sol)

		tspan = step_size .+ tspan
		u 	  = get_u(exp)
		aux = get_state(u[1:exp.n])

		(aux |> reward, aux)
	end

	function reset!() 
		@info "Reseting Env"
		u     = copy(u0)
		exp(Experiment())
		tspan = (0., step_size)
		u[1:exp.n] |> get_state
	end

	function is_end()
		time = episode_size <= tspan[1] / step_size
		alive = foldr((x,y) -> x&&y, (>).(0.1, u[1:exp.n]);init=true)
		if alive
			@show "kill bacterias"
		end
		return time || alive
	end

	exp, step!, reset!, is_end
end

