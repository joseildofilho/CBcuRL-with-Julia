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

function main()

	parsed_arguments = parse_commandline()

	params = load_config("./params/simple_auxotroph.json")

	gr()

	env = build(params["envoriment"])

	train_params = params["train"]
	q_size= train_params["Q_size"]

	learning_methods = map(collect(params["methods"])) do m
			Symbol(m.first) => m.second |> dictstring2symbol
		end

	#aux = learning_methods .|> first
	#for method in aux
	#	replace(String(method), "!" => "") * ".jl" |> include
	#end
	
	for method in learning_methods
		exper_params = build_experiment(actions(),
									reward,
									env,
									[train_params["u0"]...],
									[train_params["p"]...]
									)

		Q = Dict((i,j) => rand(2) for j in 1:q_size for i in 1:q_size)

		@info method
		f = getfield(Main, method.first)
		step!, reset!, is_end = exper_params[2:end]
		rewards = f(Q, step!, reset!, is_end; method.second...)

		exp = exper_params[1]
		plot(
			 plot(exp),
			 plot(transpose(hcat(exp.N...))),
			 plot(rewards["reward"]),
			 layout=(3,1),
			 size=(5000,1000)
		 )
		png("$(method.first)_plots.png")
	end

	#rewards = n_step_sarsa!(Q, params[2:end]..., episodes=2)
	
	
end
main()


