include("utils.jl")

function q!(Q::Dict,
			step!::Function,
			reset!::Function,
			is_end::Function;
			α::Real=1,
			ε::Real=1,
			γ::Real=0.5,
			episodes::Integer=10_000,
			αdecay ::Bool=true,
			αmin   ::Real=0.01,
			αfactor::Real=0.999,
			holdαbefore::Real=0.05,
			holdαafter ::Real=0.03,
			εdecay ::Bool=true,
			εmin   ::Real=0.001,
			εfactor::Real=0.999,
			holdεbefore::Real=0.05,
			holdεafter ::Real=0.03,
			γdecay ::Bool=false,
			γmin   ::Real=0.05,
			γfactor::Real=0.999,
			holdγbefore::Real=0.05,
			holdγafter ::Real=0.03)
	
	if αdecay
		αfactor = (αmin/α)^(1/(episodes-episodes*holdαafter))
	end
	if εdecay
		εfactor = (εmin/ε)^(1/(episodes-episodes*holdεafter))
	end
	if γdecay
		γfactor = (γmin/γ)^(1/(episodes-episodes*holdγafter))
	end

	reward_list::Array{Real,1} = []
	lr_list    ::Array{Real,1} = []
	greedy_list::Array{Real,1} = []
	for epi in 1:episodes
		state = reset!()
		reward_m = 0
		reward_c = 1
		while !is_end()
			action = e_greedy(Q[state], ε)
			R, S_ = step!(action)
			Q[state][action] += α * (R + γ * max(Q[S_]...) - Q[state][action])
			state = S_
			reward_m += R
			reward_c += 1
		end
		if ε > εmin &&
			epi > (episodes * holdεbefore) &&
			epi < (episodes * (1 - holdεafter)) &&
			εdecay
			ε *= εfactor
		end
		if α > αmin &&
			epi > (episodes * holdαbefore) &&
			epi < (episodes * (1 - holdαafter)) &&
			αdecay
			α *= αfactor
		end
		if γ > γmin &&
			epi > (episodes * holdγbefore) &&
			epi < (episodes * (1 - holdγafter)) &&
			γdecay
			γ *= γfactor
		end
		reward_result = reward_m / reward_c
		#@info reward_result ε α γ
		push!(reward_list, reward_result)
		push!(lr_list, α)
		push!(greedy_list, ε)

		print("\repisode $epi / $episodes $(round(reward_result, sigdigits=3)) α=$(round(α, sigdigits=3))  ε=$(round(ε, sigdigits=3)) γ=$(round(γ, sigdigits=3))")
	end
	Dict("reward" => reward_list,
		 "lr" => lr_list,
		 "e_greedy" => greedy_list)

end

