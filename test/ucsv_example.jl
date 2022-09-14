include(joinpath(pwd(),"src/sequential_monte_carlo.jl"))

using .SequentialMonteCarlo
using Printf,Random,Distributions,LinearAlgebra
using StaticArrays
using Plots

ucsv_prior = Uniform(0.0,1.0)

ucsv_mod(θ) = StateSpaceModel(
    UCSV(θ,0.05,(0.0,0.0)),
    (3,1)
)

T = 1000

ucsv_sim = ucsv_mod(0.2)
x,y = simulate(ucsv_sim,T)

ucsv_smc² = SMC(512,512,ucsv_mod,ucsv_prior,3,0.5)
smc²(ucsv_smc²,y)

plt = @animate for t in 2:T
    smc²!(ucsv_smc²,y,t)
    histogram(
        ucsv_smc².θ,
        weights = ucsv_smc².ω,
        normalize = :pdf,
        xlabel = "γ",
        grid = false,
        tickdir = :out,
        xlim = (0.0,1.0),
        ylim = (0.0,20.0),
        framestyle = :box,
        legend = false
    )
    plot!(plot_title=string("UCSV at t = ",t))
end every 10

gif(plt,"ucsv_animation.gif",fps=15)