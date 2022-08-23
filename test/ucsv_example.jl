include(joinpath(pwd(),"src/sequential_monte_carlo.jl"))

using .SequentialMonteCarlo
using Printf,Random,Distributions,LinearAlgebra
using StaticArrays

ucsv_prior = TupleProduct((
    Uniform(0.0,1.0)
))

ucsv_mod(θ) = StateSpaceModel(
    UCSV(θ[1],0.05,(0.0,0.0)),
    (3,1)
)

ucsv_sim = ucsv_mod([0.2])
x,y = simulate(ucsv_sim,1000)

ucsv_smc² = SMC(Random.GLOBAL_RNG,512,512,ucsv_mod,ucsv_prior,3,0.5)