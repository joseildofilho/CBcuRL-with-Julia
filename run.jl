include("dictstring2symbol.jl")
include("experiment.jl")
include("system_equations.jl")
include("single_auxotroph.jl")
include("q.jl")
include("n_step_sarsa.jl")
include("random_method.jl")
include("n_step_off_policy_sarsa.jl")

using DifferentialEquations
using Plots
using ArgParse
using JSON
using InteractiveUtils
using Base.Cartesian
using Dates

function parse_commandline()
	s = ArgParseSettings()

	@add_arg_table s begin 
		"config"
			help = "The file used to configure the project"
			arg_type = String
			required = true
		"results_path"
			help = "Set folder where the files goes"
			arg_type = String
			default = "results"
	end
	parse_args(s)
end

load_config(path::String) = path |> open |> JSON.parse

function main(;path::String = "")

	parsed_arguments = parse_commandline()

	params = (path == "" ? parsed_arguments["config"] : path) |> load_config

	gr()

	env = build(params["envoriment"])

	train_params = params["train"]

	learning_methods = map(collect(params["methods"])) do m
			Symbol(m.first) => m.second |> dictstring2symbol
		end

	experiment_time = Dates.format(now(), "HH_MM_d_u_y")

	#aux = learning_methods .|> first
	#for method in aux
	#	replace(String(method), "!" => "") * ".jl" |> include
	#end
	
	@info learning_methods
	for method in learning_methods
		actions_list = build_actions(train_params["Cin"])
		aux1::Array{Array, 1} = train_params["bounds"]
		aux2::Array = train_params["rewards"]
		reward = build_reward(aux1, aux2)

		experiments::Array{Experiment, 1} = []
		rewards::Array{Array{Real, 1},1}  = []
		durations::Array{Array{Integer, 1}, 1} = []
		e = 0

		for i in 1:10
			exper_params = build_experiment(actions_list,
									reward,
									env,
									[train_params["u0"]...],
									[train_params["p"]...];
									number_species=params["envoriment"]["species"],
									upper_bound=train_params["upper_bound"]
									)

			Q = Dict(i.I =>	rand(length(actions_list))
				for i in CartesianIndices(Tuple(
						  [train_params["Q_size"] 
						   for _ in 1:params["envoriment"]["species"]])))

			@info method
			f = getfield(Main, method.first)
			step!, reset!, is_end = exper_params[2:end]
			method_return = f(Q, step!, reset!, is_end; method.second...)

			exp = exper_params[1]
			push!(experiments, exp)
			push!(rewards, method_return["reward"])
			push!(durations, method_return["duration"])
			e = method_return["e_greedy"]
			
		end
		med_rewards = (foldr(rewards) do list, acc
				   if length(list) == 0
					   list
				   else
					   acc + list
				   end
			   end)

		mean_duration = foldr(durations) do list, acc
					if length(list) == 0
						list
					else
						acc + list
					end
				end

		@show med_rewards[1]

		path = "$(parsed_arguments["results_path"])/$(experiment_time)"
		mkpath(path)
		
		args = Dict(:size=>(900,900), :linewidth=>3)
		plot(med_rewards, ylim=(0,15), xlim=(0,150); args...)
		png("$path/$(method.first)_mean_rewards.png")

		plot(mean_duration, ylim=(0,15), xlim=(0,150); args...)
		png("$path/$(method.first)_mean_duration.png")

		plot(e, ylim=(0,1), xlim=(0,150); args...)
		png("$path/$(method.first)_epsilon.png")

		plot(experiments[end], ylim=(0,15), xlim=(0,150); args...)
		png("$path/$(method.first)_some_time.png")

		return (exp=experiments, rewards=rewards, e=e)
	end

	#rewards = n_step_sarsa!(Q, params[2:end]..., episodes=2)
	
	
end
main()


