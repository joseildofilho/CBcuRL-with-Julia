include("dictstring2symbol.jl")
include("experiment.jl")
include("system_equations.jl")
include("single_auxotroph.jl")
include("q.jl")
include("n_step_sarsa.jl")

using DifferentialEquations
using Plots
using ArgParse
using JSON
using InteractiveUtils
using Base.Cartesian

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

	learning_methods = map(collect(params["methods"])) do m
			Symbol(m.first) => m.second |> dictstring2symbol
		end

	#aux = learning_methods .|> first
	#for method in aux
	#	replace(String(method), "!" => "") * ".jl" |> include
	#end
	
	for method in learning_methods
		actions_list = build_actions(train_params["Cin"])
		aux1::Array{Array, 1} = train_params["bounds"]
		aux2::Array = train_params["rewards"]
		reward = build_reward(aux1, aux2)
		exper_params = build_experiment(actions_list,
									reward,
									env,
									[train_params["u0"]...],
									[train_params["p"]...]
									)

		#Q = Dict((i,j) => rand(2) for j in 1:q_size for i in 1:q_size)
		Q = zeros([train_params["Q_size"] 
				   for i in 1:params["envoriment"]["species"]]...)
		Q = Dict(i.I 
				 =>
				rand(length(actions_list))
				 for i in CartesianIndices(Q))

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


