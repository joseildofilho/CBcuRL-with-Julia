include("./grid_world.jl")
include("./utils.jl")
function main()
	grid = simple_grid()

	Q = Dict(state => ((grid.states[state] == STATE) ? randn(length(grid.actions)) : zeros(length(grid.actions)))
			 for state in 1:grid.nS)

	e 	  = 0.9
	alpha = 0.9
	n 	  = 2
	gamma = 0.9


	state_history  = History()
	action_history = History()
	reward_history = History()

	e_greedy = build_e_greedy(Q, grid.nA)

	policy = greedy(Q)

	episodes = 1:100000

	for episode in episodes
		print("\repisodes: $episode / $episodes $(Q[6])\r")
		reset.([state_history, action_history, reward_history])
		S = reset!(grid)
		state_history(S)

		A = policy[S]
		action_history(A)

		T = Inf
		t = 0
		tau = 0
		terminal = false
		while !(tau == T - 1)
			if t < T
				S, R, terminal = step!(grid, A)
				state_history(S)
				reward_history(R)
				if terminal
					T = t + 1
				else
					A = policy[S]
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
				policy[state_tau] = e_greedy(state_tau, e)
			end
			t += 1
		end
		if episode < 500
			alpha *= 0.99
#			gamma *= 0.9999
			e *= 0.99
		end
	end

	policy = greedy(Q)
	show(grid)

	show_policy(policy)
	for i in 1:15
		@show Q[i], i, argmax(Q[i])
	end
end
main()
