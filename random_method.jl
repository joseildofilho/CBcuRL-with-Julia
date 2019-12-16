include("q.jl")
function random_method(Q::Dict,
		 step!::Function,
		 reset!::Function,
		 is_end::Function;
		 kwargs ...)
	q!(Q, step!, reset!, is_end; 
	   ε=1,
	   εdecay=false,
	   episodes=kwargs[Symbol("episodes")])
end
