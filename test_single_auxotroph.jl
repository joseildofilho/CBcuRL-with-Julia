include("single_auxotroph.jl")

using Test

function test1()
	bounds::Array{Array, 1} = [[2,2], [10,10]]
	rewards::Array = [1.0,-1.0]
	reward::Function = build_reward(bounds, rewards)

	point::Tuple = (1,1)
	reward(point) == -1

	point = (2,2)
	reward(point) == 1

	point = (2,10)
	reward(point) == 1

	point = (10,2)
	reward(point) == 1

	point = (10,10)
	reward(point) == 1

	point = (11,11)
	reward(point) == -1

	point = (11,2)
	reward(point) == -1

	point = (2,11)
	reward(point) == -1

	point = (10,1)
	reward(point) == -1

	point = (1,10)
	reward(point) == -1
end
@test test1()
