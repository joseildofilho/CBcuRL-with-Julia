include("utils.jl")

function n_step_sarsa!(Q::Dict, 
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


	n 	  = 2

	state_history::History  = History()
	action_history::History = History()
	reward_history::History = History()

	reward_mean_history::Array{Real, 1} = []
	lr_history 		   ::Array{Real, 1} = []
	greedy_history 	   ::Array{Real, 1} = []

	for episode in 1:episodes
		println("\repisodes: $episode / $episodes")

		reward_mean::Integer  = 0
		reward_count::Integer = 0

		reset.([state_history, action_history, reward_history])
		S::Tuple = reset!()
		state_history(S)

		A::Integer = e_greedy(Q[S],ε)
		action_history(A)

		T::Real = Inf
		t::Integer = 0
		tau::Integer = 0
		while !(tau == T - 1)
			if t < T
				R::Real, S = step!(A)

				reward_mean  += R
				reward_count += 1

				state_history(S)
				reward_history(R)
				if is_end()
					T = t + 1
				else
					A = e_greedy(Q[S],ε)
					action_history(A)
				end
			end
			tau = t - n + 1
			if tau >= 0
				Rs 	   = reward_history.hist[1:end]
				γs = [γ ^ (i - 1) for i in 1:length(Rs)]
				G = sum((*).(Rs, γs))
				reward_history()
				if tau + n < T
					state_aux  = state_history.hist[end]
					action_aux = action_history.hist[end]
					G = G + (γ ^ n) * Q[state_aux][action_aux]
				end
				state_tau  = state_history()
				action_tau = action_history()
				Q[state_tau][action_tau] += α * (G - Q[state_tau][action_tau])
			end
			t += 1
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
		@info reward_mean/reward_count ε α γ
		push!(lr_history, α)
		push!(greedy_history, ε)
		push!(reward_mean_history, reward_mean / reward_count)
	end
	Dict("reward" => reward_mean_history,
		 "lr" => lr_history,
		 "e_greedy" => greedy_history)
end
