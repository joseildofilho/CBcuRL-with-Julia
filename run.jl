include("dictstring2symbol.jl")
include("experiment.jl")
include("simple_auxotroph.jl")
include("q.jl")
include("n_step_sarsa.jl")

using DifferentialEquations
using Plots
using ArgParse
using JSON
using InteractiveUtils

function parse_commandline()
	s = ArgParseSettings()

	@add_arg_table s begin 
		"config"
			help = "The file used to configure the project"
			arg_type = String
			required = true
	end

	parse_args(s)
end

load_config(path::String) = path |> open |> JSON.parse

function import_functions(methods::Array)
	for method in methods
		replace(method, "!" => "") |> include
	end
end

function main()

	parsed_arguments = parse_commandline()

	params = load_config("./params/simple_auxotroph.json")

	gr()

	env = build(params["envoriment"])
	methods = map(collect(params["methods"])) do m
		Symbol(m.first) => m.second |> dictstring2symbol
		end


	exper_params = build_experiment(actions(),
									reward,
									step_size=3,
									number_species=2,
									env)

	size_q = 10
	Q = Dict((i,j) => rand(2) for j in 1:size_q for i in 1:size_q)


	for method in methods
		rewards = getfield(Main, method.first)(Q, exper_params[2:end]... 
											   ;method.second...)
	end

	#rewards = n_step_sarsa!(Q, params[2:end]..., episodes=2)
	exp = exper_params[1]
	
	plot(
		 plot(exp),
		 plot(transpose(hcat(exp.N...))),
		 plot(rewards["reward"]),
		 layout=(3,1),
		size=(5000,1000)
		 )
	png("plots.png")
	
	Dict("rewards" => rewards, 
		 "Q" => Q,
		 "params" => exp_params,
		 )
end
main()


