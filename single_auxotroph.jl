function build_actions(inputs::Array)
	actions::Array{Function, 1} = []
	num_actions = 2 ^ length(inputs)
	for action in 0:num_actions-1
		push!(actions,
			function(p)
				p[2:end] = (*).(digits(action, base=2, pad=length(inputs)),
								inputs)
				p
			end
			)
	end
	actions
end

function build_reward(box::Array{Array,1}, reward::Array)
	positive_side(point::Tuple) = foldr((x,y)->x&&y, (>=).(point, box[1]);
										init=true)
	negative_side(point::Tuple) = foldr((x,y)->x&&y, (<=).(point, box[2]);
										init=true)
	(state::Tuple) -> if positive_side(state) && negative_side(state)
		reward[1]
	else
		reward[2]
	end
end


