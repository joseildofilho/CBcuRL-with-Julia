include("experiment.jl")
include("simple_auxotroph.jl")
include("q.jl")

using DifferentialEquations
using Plots

gr()

function main()

	params = exp, step!, reset!, is_end! = build_experiment(actions(),
															reward,
															step_size=3)

	size_q = 10
	Q = Dict((i,j) => rand(2) for j in 1:size_q for i in 1:size_q)

	q!(Q, params[2:end]...)
	
	plot(
		 plot(exp),
		 plot(transpose(hcat(map(x->x[3:4], exp.U)...))),
		 layout=(2,1),
		size=(1700,1000)
		 )
	png("plots.png")

end
main()


