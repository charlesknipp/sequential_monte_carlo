include(joinpath(pwd(),"src/sequential_monte_carlo.jl"))

using .SequentialMonteCarlo
using Printf,Random,Distributions,LinearAlgebra,Plots

function plot_histograms(smc::SMC)
    θ   = reduce(hcat,smc.θ)
    plt = plot(
        histogram(θ[1,:],weights=smc.ω,normalize=:probability,xlabel="A"),
        histogram(θ[2,:],weights=smc.ω,normalize=:probability,xlabel="Q"),
        histogram(θ[3,:],weights=smc.ω,normalize=:probability,xlabel="R"),
        legend=false
    )

    return plt
end

## SETUP & SIMULATION #########################################################

test_prior = product_distribution([
    TruncatedNormal(0,1,-1,1),
    LogNormal(),
    LogNormal()
])

mod_func(θ) = StateSpaceModel(
    LinearGaussian(θ[1],1.0,θ[2],θ[3],0.0),
    (1,1)
)

test_θ = [0.5,1.0,1.0]
x,y = simulate(mod_func(test_θ),1000)

## DENSITY TEMPERED ###########################################################

smc_sampler = SMC(512,1024,mod_func,test_prior,3,0.5)
density_tempered(smc_sampler,y[1:250])

# save histograms after dt estimation
jump_start_plot1 = plot_histograms(smc_sampler)
savefig(jump_start_plot1,"visuals/jump_start_1.png")
savefig(jump_start_plot1,"visuals/jump_start_1.svg")

## JUMP START SMC² ############################################################

for m in 1:smc_sampler.M
    likelihood,smc_sampler.w[m],_ = particle_filter!(
        smc_sampler.x[m],
        smc_sampler.w[m],
        y[251],
        smc_sampler.model(smc_sampler.θ[m]),
        nothing
    )

    smc_sampler.ω[m]     = likelihood
    smc_sampler.logZ[m] += likelihood
end

# reweight particles
_,smc_sampler.ω,smc_sampler.ess = SequentialMonteCarlo.normalize(smc_sampler.ω)

for t in 252:1000
    smc²!(smc_sampler,y,t)
end

# save histograms after jump starting smc²
jump_start_plot2 = plot_histograms(smc_sampler)
savefig(jump_start_plot2,"visuals/jump_start_2.png")
savefig(jump_start_plot2,"visuals/jump_start_2.svg")