using ModelingToolkit

function build()
	@parameters t q γ1 γT C0in CTin
	@variables N1(t) N2(t) C0(t) CT(t)
	@derivatives D'~t

	Ks1 = 6.845928 * (10 ^ (-5))
	KT  = 1.02115 * (10 ^ (-7))

	μmax1 = 1.5
	μmax2 = 3

	μ1 = (C0)  	  -> μmax1 * (C0/(Ks1 + C0))
	μ2 = (C0, CT) -> μmax2 * (C0/(KT + C0))

	eqs = [
		   D(N1) ~ N1*(μ1(C0) - q),
		   D(N2) ~ N2*(μ2(C0, CT) - q),
		   D(C0) ~ q*(C0in - C0) - (1/γ1) * μ1(C0)*N1,
		   D(CT) ~ q*(CTin - CT) - (1/γT)*μ2(C0,CT)*N2
		   ]

	de = ODESystem(eqs)

	f  = ODEFunction(de, [N1, N2, C0, CT], [q, γ1, γT, C0in, CTin])

	u0    = [10000., 100000., 100., 0.]
	tspan = (0., 5.)
	p     = [0.5, 4.8*10^12, 5.2*10^12, 1.5, 3, 0.25, 0]

	(f, u0, tspan, p)
end
