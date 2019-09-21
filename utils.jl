function build_e_greedy(Q_function, actions)
	(state, e) -> rand() > e ? argmax(Q_function[state]) : rand(1: actions)
end

function greedy(Q)
	x = []
	for i in 1:length(Q)
		insert!(x, 1, argmax(Q[i]))
	end
	x
end

struct History
	hist::Array{Any, 1}
	History(n::Int32) = new([nothing for _ in 1:n])
	History() = new([])
end

function (h::History)(item)
	append!(h.hist, item)
#	popfirst!(h.hist)
end

(h::History)() = popfirst!(h.hist)

reset(h::History) = empty!(h.hist)

function show_policy(policy)
	keys = Dict(1=>"U", 2=>"R", 3=>"D", 4=>"L")
	for (i, x) in enumerate(policy)
		print("$(keys[x]) ")
		if i % 4 == 0
			print("\n")
		end
	end
	print("\r\n")
end
