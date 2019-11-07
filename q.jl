include("utils.jl")

function q!(Q::Dict,
			step!::Function,
			reset!::Function,
			is_end::Function;
			α::Real=1,
			ε::Real=1,
			γ::Real=0.5,
			episodes::Integer=10_000,
			αmin   ::Real=0.01,
			αfactor::Real=0.999,
			εmin   ::Real=0.001,
			εfactor::Real=0.999,
			γmin   ::Real=0.05,
			γfactor::Real=0.999)

	reward_list::Array{Real,1} = []
	lr_list    ::Array{Real,1} = []
	greedy_list::Array{Real,1} = []
	for epi in 1:episodes
		state = reset!()
		reward_m = 0
		reward_c = 0
		@info "episode $epi / $episodes"
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
		if α > αmin
			α *= αfactor
		end
		if γ > γmin
			γ *= γfactor
		end
		reward_result = reward_m / reward_c
		@info reward_result ε α γ
		push!(reward_list, reward_result)
		push!(lr_list, α)
		push!(greedy_list, ε)
	end
	Dict("reward" => reward_list,
		 "lr" => lr_list,
		 "e_greedy" => greedy_list)

end

