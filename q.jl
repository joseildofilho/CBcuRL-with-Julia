include("utils.jl")

function q!(Q::Dict,
			step!::Function,
			reset!::Function,
			is_end::Function;
			α::Real=0.333,
			ε::Real=0.9,
			γ::Real=0.9,
			episodes::Integer=10_000,
			εmin=0.1,
			εfactor=0.9)

	reward_list::Array{Real,1} = []
	for epi in 1:episodes
		state = reset!()
		reward_m = 0
		reward_c = 0
		@info "episode $epi"
		while !is_end()
			action = e_greedy(Q[state], ε)
			R, S_ = step!(action)
			Q[state][action] += α * (R + γ * max(Q[S_]...) - Q[state][action])
			state = S_
			reward_m += R
			reward_c += 1
		end
		if ε > εmin
			ε *= εfactor
		end
		reward_result = reward_m / reward_c
		@show reward_result
		append!(reward_list, reward_result)
	end
	Dict("reward" => reward_list)

end

