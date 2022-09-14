include(joinpath(pwd(),"src/sequential_monte_carlo.jl"))

using .SequentialMonteCarlo
using Printf,Random,Distributions,LinearAlgebra,Plots
using Plots.PlotMeasures


function plot_histograms(smc::SMC,labels::AbstractVector)
    θ = reduce(hcat,smc.θ)
    plots = map(
        i -> histogram(
            θ[i,:],
            weights = smc.ω,
            normalize = :probability,
            xlabel = labels[i],
            ylim = 0.3,
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

lg_prior = product_distribution([
    TruncatedNormal(0,1,-1,1),
    LogNormal(),
    LogNormal()
])

lg_mod(θ) = StateSpaceModel(
    LinearGaussian(θ[1],1.0,θ[2],θ[3],0.0),
    (1,1)
)

T = 1000
lg_θ = [0.5,1.0,1.0]
x,y  = simulate(lg_mod(lg_θ),T)


## density tempered
lg_dt_smc = SMC(512,1024,lg_mod,lg_prior,3,0.5)
density_tempered(lg_dt_smc,y)

expected_parameters(lg_dt_smc)
plot_histograms(lg_dt_smc,["A","Q","R"])


## smc²
lg_smc² = SMC(512,1024,lg_mod,lg_prior,3,0.5)
smc²(lg_smc²,y)

plt = @animate for t in 2:T
    smc²!(lg_smc²,y,t)
    plot_histograms(lg_smc²,["A","Q","R"])
    plot!(plot_title="\n"*t)
end every 100



gif(plt,"linear_gaussian_animation.gif",fps=15)
expected_parameters(lg_smc²)
plot_histograms(lg_smc²,["A","Q","R"])
plot!(plot_title="\n5000")