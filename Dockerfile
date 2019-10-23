FROM julia

RUN apt-get update && apt-get install -y build-essential

RUN julia -e 'using Pkg; \
		Pkg.add(["DifferentialEquations", \
				"Plots", \
				"ModelingToolkit", \
				"LSODA", \
				"Crayons" \
		]); '

COPY . /apt/source-code

RUN ls /apt/source-code

ENTRYPOINT julia /apt/source-code/experiment_1.jl
