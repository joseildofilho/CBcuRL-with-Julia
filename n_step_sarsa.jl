include("utils.jl")

function n_step_sarsa!(Q::Dict, 
					  step!::Function,
					  reset!::Function,
					  is_end::Function;
					  episodes=10_000)

	e 	  = 0.9
	alpha = 0.9
	n 	  = 2
	gamma = 0.9

	state_history::History  = History()
	action_history::History = History()
	reward_history::History = History()

	reward_mean_history::Array{Real,1} = []

	for episode in 1:episodes
		println("\repisodes: $episode / $episodes")

		reward_mean::Integer  = 0
		reward_count::Integer = 0

		reset.([state_history, action_history, reward_history])
		S::Tuple = reset!()
		state_history(S)

		A::Integer = e_greedy(Q[S],e)
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
					A = e_greedy(Q[S],e)
					action_history(A)
				end
			end
			tau = t - n + 1
			if tau >= 0
				Rs 	   = reward_history.hist[1:end]
				gammas = [gamma ^ (i - 1) for i in 1:length(Rs)]
				G = sum((*).(Rs, gammas))
				reward_history()
				if tau + n < T
					state_aux  = state_history.hist[end]
					action_aux = action_history.hist[end]
					G = G + (gamma ^ n) * Q[state_aux][action_aux]
				end
				state_tau  = state_history()
				action_tau = action_history()
				Q[state_tau][action_tau] += alpha * (G - Q[state_tau][action_tau])
			end
			t += 1
		end
		if e > 0.1
			alpha *= 0.99
			e *= 0.99
		end
		@show reward_mean/reward_count
		push!(reward_mean_history, reward_mean / reward_count)
	end
	@show reward_mean_history
	Dict("reward" => reward_mean_history)	
end
