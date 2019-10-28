# How to Configure the experiments.

## The Json

```
	{
		"envoriment": ...
	}
```

### Configuring the Envoriment
```
	{
		"envoriment": {
			"species" : number of species #[Integer],
			"concentrations" : number of concentrations, substracts. #[Integer],
			"q" : Diluition rate #[Float],
			"γ0" : The yield coefficients for bacterial populations on the com
			mom source #[Float],
			"γ" : The yield coefficient for each population, based on each 
			auxotroph trigger, 0.0 means that population doenst use a auxotroph
			trigger #[Float],
			"K" : The half-speed constant for each specie in a givem mean,
			each line of the matrix represents a vector of k's for a given 
			specie, values 0.0 means that population it isnt affect for auxotroph
			tigger #[Array[Array[Float]]],
			"μmax" : vector of max μ for each specie
		}
	}
```

**P.S.: Examples in ./params/**
