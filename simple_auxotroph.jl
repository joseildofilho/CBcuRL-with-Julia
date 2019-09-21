include("grid_world.jl")

using ModelingToolkit

function inject_0!(p)
	p[end] = 0.
	p
end

function inject_value!(p)
	p[end] = 0.333
	p
end

actions() = [
			 p -> inject_0!(p),
			 p -> inject_value!(p)
			 ]

function build()
	@parameters C0in CTin t
	@variables N1(t) N2(t) C0(t) CT(t)
	@derivatives D'~t

	q = .5

	γ1 = 4.8*10^12
	γT = 5.2*10^12

	Ks1 = 6.845928 * (10 ^ (-5))
	KT  = 1.02115 * (10 ^ (-7))

	μmax1 = 1.5
	μmax2 = 3

	μ1 = μmax1 * (C0/(Ks1 + C0))
	μ2 = μmax2 * (C0/(Ks1 + C0))*(CT/(KT + CT))

	eqs = [
		   D(N1) ~ N1*(μ1 - q),
		   D(N2) ~ N2*(μ2 - q),
		   D(C0) ~ q*(C0in - C0) - ((1/γ1) * μ1 * N1) - ((1/γ1) * μ2 * N2),
		   D(CT) ~ q*(CTin - CT) - (1/γT) * μ2 * N2
		   ]

	de = ODESystem(eqs)

	f  = ODEFunction(de, [N1, N2, C0, CT], [C0in, CTin])

	u0    = [1000000., 1000000., 100., 0.]
	p     = [0.25, 
			 0.]

	(f, u0, p)
end
