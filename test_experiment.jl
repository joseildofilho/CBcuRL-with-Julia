using Test

include("experiment.jl")

############### TESTING GET_STATE ###############

basic_state = []

@test_throws ArgumentError get_state(basic_state)

basic_state = [1e6, 1e6]
@test_throws ArgumentError get_state(basic_state, n=0)

basic_state = [1e6, 1e6]
@test_throws ArgumentError get_state(basic_state, n=-1)

basic_state = [-1e6, 1e6]
@test_throws ArgumentError get_state(basic_state, n=-1)

basic_state = [1e6, -1e6]
@test_throws ArgumentError get_state(basic_state, n=-1)

basic_state = [-1e6, -1e6]
@test_throws ArgumentError get_state(basic_state, n=-1)

basic_state = [1e6, 1e6]
@test_throws ArgumentError get_state(basic_state, factor=0.0)

basic_state = [1e6, 1e6]
@test get_state(basic_state) == (1, 1)

basic_state = (*).(2, basic_state)
@test get_state(basic_state) == (2, 2)

basic_state = (*).(5, basic_state)
@test get_state(basic_state) == (10, 10)

basic_state = (*).(5, basic_state)
@test get_state(basic_state) == (10, 10)

basic_state = [1e6, 1e6, 1e5]
@test get_state(basic_state) == (1, 1, 0)

############### TESTING GET_STATE ###############
