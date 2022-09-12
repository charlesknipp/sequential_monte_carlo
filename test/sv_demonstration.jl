include(joinpath(pwd(),"src/sequential_monte_carlo.jl"))

using .SequentialMonteCarlo
using Printf,Random,Distributions,LinearAlgebra,Plots
using Plots.PlotMeasures

#=
function plot_histograms(smc::SMC,labels::AbstractVector)
    θ   = reduce(hcat,smc.θ)
    plt = plot(
        histogram(θ[1,:],weights=smc.ω,normalize=:probability,xlabel=labels[1]),
        histogram(θ[2,:],weights=smc.ω,normalize=:probability,xlabel=labels[2]),
        histogram(θ[3,:],weights=smc.ω,normalize=:probability,xlabel=labels[3]),
        legend=false
    )

    return plt
end
=#

function plot_histograms(smc::SMC,labels::AbstractVector)
    θ = reduce(hcat,smc.θ)
    plots = map(
        i -> histogram(
            θ[i,:],
            weights = smc.ω,
            normalize = :probability,
            xlabel = labels[i],
            grid = false,
            tickdir = :out,
            bins = 25
        ),
        1:length(labels)
    )

    return plot(
        plots...,
        legend = false,
        layout = (1,length(labels)),
        size = (1200,500),
        framestyle = :box,
        top_margin = 15mm,
        margin = 10mm,
        bottom_margin = 10mm
    )
end

sv_prior = product_distribution([
    Normal(0.0,2.0),
    Uniform(-1.0,1.0),
    Gamma(1.0,1.0)
])

sv_mod(θ) = StateSpaceModel(
    StochasticVolatility(θ...),
    (1,1)
)

T = 5000

θ = [-1.0,0.9,0.2]
x,y = simulate(sv_mod(θ),T)


## density tempered
sv_dt_smc = SMC(Random.GLOBAL_RNG,512,1024,sv_mod,sv_prior,3,0.5)
dt_plot_1 = plot_histograms(sv_dt_smc,["μ","ρ","σ"])

density_tempered(sv_dt_smc,y)

expected_parameters(sv_dt_smc)
dt_plot_2 = plot_histograms(sv_dt_smc,["μ","ρ","σ"])


## smc²
sv_smc² = SMC(Random.GLOBAL_RNG,512,512,sv_mod,sv_prior,3,0.5)
smc²(sv_smc²,y)


plt = @animate for t in 2:T
    smc²!(sv_smc²,y,t)
    plot_histograms(sv_smc²,["μ","ρ","σ"])
    plot!(plot_title=string("\n",t))
end every 10

gif(plt,"stochastic_volatility_animation.gif",fps=15)
expected_parameters(sv_smc²)
smc²_plot_2 = plot_histograms(sv_smc²,["μ","ρ","σ"])
