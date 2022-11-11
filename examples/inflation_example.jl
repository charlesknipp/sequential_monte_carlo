using SequentialMonteCarlo
using Random,Distributions,LinearAlgebra
using Plots
using FredData
using StatsBase
using Printf

#ENV["https_proxy"] = "http://wwwproxy.frb.gov:8080"
BLAS.set_num_threads(Threads.nthreads())

# get quarterly GDP using the FRED API (ecae2fc8d6c684847525a828ae7a3ab8)
pce_series = get_data(
    Fred(),
    "PCECTPI",
    observation_start = "1960-01-01",
    observation_end = "2020-01-01",
    units = "pc1",
    frequency = "q"
)

pce_data = pce_series.data[:,3:4]
y = pce_data.value
T = length(y)


## Define & Run the UC Model ##################################################

uc_mod(θ) = StateSpaceModel(
    UC(θ...),
    (1,1)
)

uc_prior = product_distribution([
    Normal(3.0,2.0),
    Uniform(0.0,4.0),
    Uniform(0.0,4.0)
])

function get_quantiles_uc(smc::SMC,yt::Float64)
    xquantiles = zeros(Float64,smc.M,3)
    cquantiles = zeros(Float64,smc.M,3)
    variances  = zeros(Float64,smc.M)
    
    for i in 1:smc.M
        xquantiles[i,:] .= quantile(smc.x[i],weights(smc.w[i]),[0.25,0.5,0.75])
        cquantiles[i,:] .= quantile(yt.-smc.x[i],weights(smc.w[i]),[0.25,0.5,0.75])
        variances[i,:]  .= var(smc.x[i],weights(smc.w[i]),)
    end

    return (
        vec(mean(xquantiles,weights(smc.ω),1)),
        vec(mean(xquantiles,weights(smc.ω),1)),
        mean(variances,weights(smc.ω))
    )
end

Random.seed!(1998)
uc_smc² = SMC(1024,512,uc_mod,uc_prior,3,0.5)

uc_xqs = fill(zeros(Float64,3),T)
uc_cqs = fill(zeros(Float64,3),T)
uc_vars = zeros(Float64,T)

smc²(uc_smc²,y)
debug_list = []

for t in 2:T
    uc_xqs[t-1],uc_cqs[t-1],uc_vars[t-1] = get_quantiles_uc(uc_smc²,y[t-1])
    smc²!(uc_smc²,y,t)
    #if (t > 193) & (uc_smc².acc_ratio < 0.005)
    #    smc_copy = deepcopy(uc_smc²)
    #    push!(debug_list,smc_copy)
    #end
end

# debugging over degenerate samples
for smc in debug_list
    @printf("ess = %4.3f\t",smc.ess)

    θ = hcat(smc.θ...)
    ω = smc.ω

    Σ = cov(θ')
    #print(stdout,"\ncov(θ) = ")
    #Base.show(stdout,"text/plain",Σ)
    @printf("norm(cov(θ)) = %3.7f   ",det(Σ))

    Σ = cov(θ,weights(ω),2)
    #print(stdout,"\nwcov(θ) = ")
    #Base.show(stdout,"text/plain",Σ)
    @printf("norm(wcov(θ)) = %3.7f",det(Σ))
    
    print("\n")
end

uc_xqs[T],uc_cqs[T],uc_vars[T] = get_quantiles_uc(uc_smc²,y[T])

## Plot Results ###############################################################

uc_plt1 = scatter(
    pce_data.value,
    xformatter = x -> pce_data.date[Int(x+1)],
    framestyle = :box,
    color = :black,
    label = "observed data"
)

plot!(
    hcat(uc_xqs...)[1,:],
    fillrange = hcat(uc_xqs...)[3,:],
    fillcolor = :grey,
    fillalpha = 0.35,
    linealpha = 0.0,
    label = false
)

plot!(
    hcat(uc_xqs...)[2,:],
    color = :red,
    ylims = (-5,15),
    label = "filtered trend (UC)",
    title = "quarterly PCE inflation rate"
)

uc_plt2 = plot(
    hcat(uc_cqs...)[1,:],
    xformatter = x -> pce_data.date[Int(x+1)],
    fillrange = hcat(uc_cqs...)[3,:],
    fillcolor = :grey,
    fillalpha = 0.5,
    linealpha = 0.0,
    label = false
)

plot!(
    hcat(uc_cqs...)[2,:],
    color = :red,
    framestyle = :box,
    ylims = (-5,5),
    label = "filtered cycle (UC)",
    title = "quarterly PCE inflation rate"
)

savefig(uc_plt1,"pce_inflation_trend_1960-2020.pdf")
savefig(uc_plt2,"pce_inflation_cycle_1960-2020.pdf")


## Run Bootstrap Filter #######################################################

pred_uc_θ = expected_parameters(uc_smc²)
pred_uc_mod = uc_mod(pred_uc_θ)

function get_latent_states_uc(
        rng::AbstractRNG,
        N::Int64,
        y::Vector{Float64},
        model::StateSpaceModel,
        proposal=nothing
    )
    xq = fill(zeros(Float64,3),length(y))
    cq = fill(zeros(Float64,3),length(y))
    vars = zeros(Float64,length(y))

    x,w,_ = particle_filter(rng,N,y[1],model,proposal)
    xq[1] = quantile(x,[0.25,0.5,0.75])
    cq[1] = quantile(y[1].-x,[0.25,0.5,0.75])

    for t in 2:length(y)
        _,w,_ = particle_filter!(rng,x,w,y[t],model,proposal)
        xq[t] = quantile(x,[0.25,0.5,0.75])
        cq[t] = quantile(y[t].-x,[0.25,0.5,0.75])
        vars[t] = var(x)
    end

    return xq,cq,vars
end

uc_pred_xqs,uc_pred_cqs,uc_pred_vars = get_latent_states_uc(Random.GLOBAL_RNG,1024,y,pred_uc_mod)

pred_uc_plt1 = scatter(
    pce_data.value,
    framestyle = :box,
    color = :black,
    label = "observed data"
)

plot!(
    hcat(uc_pred_xqs...)[1,:],
    xformatter = x -> pce_data.date[Int(x+1)],
    fillrange = hcat(uc_pred_xqs...)[3,:],
    fillcolor = :grey,
    fillalpha = 0.5,
    linealpha = 0.0,
    label = false
)

plot!(
    hcat(uc_pred_xqs...)[2,:],
    linecolor = :red,
    label = "filtered trend (UC)",
    title = "quarterly PCE inflation rate (given θ)"
)

pred_uc_plt2 = plot(
    hcat(uc_pred_cqs...)[1,:],
    xformatter = x -> pce_data.date[Int(x+1)],
    fillrange = hcat(uc_pred_cqs...)[3,:],
    fillcolor = :grey,
    fillalpha = 0.5,
    linealpha = 0.0,
    label = false
)

plot!(
    hcat(uc_pred_cqs...)[2,:],
    color = :red,
    framestyle = :box,
    ylims = (-5,5),
    label = "filtered cycle (UC)",
    title = "quarterly PCE inflation rate (given θ)"
)

savefig(pred_uc_plt1,"pce_inflation_trend_1960-2020_post.pdf")
savefig(pred_uc_plt2,"pce_inflation_cycle_1960-2020_post.pdf")


## Define & Run the UC-SV Model ###############################################

ucsv_mod(θ) = StateSpaceModel(
    UCSV(θ[1],θ[2],(θ[3],θ[4])),
    (3,1)
)

ucsv_prior = product_distribution([
    Uniform(0.0,1.0),    
    Normal(3.0,2.0),
    Uniform(0.0,2.0),
    Uniform(0.0,2.0)
])

function get_quantiles_ucsv(smc::SMC,yt::Float64)
    xquantiles = zeros(Float64,3)
    cquantiles = zeros(Float64,3)
    variance   = 0.0
    
    for i in 1:smc.M
        x_cloud = reduce(hcat,smc.x[i])[1,:]
        xquantiles += smc.ω[i]*quantile(x_cloud,[0.25,0.5,0.75])
        cquantiles += smc.ω[i]*quantile(yt.-x_cloud,[0.25,0.5,0.75])
        variance   += smc.ω[i]*var(x_cloud)
    end
    return xquantiles,cquantiles,variance
end

Random.seed!(1998)
ucsv_smc² = SMC(8192,512,ucsv_mod,ucsv_prior,5,0.5)

ucsv_xqs = fill(zeros(Float64,3),T)
ucsv_cqs = fill(zeros(Float64,3),T)
ucsv_vars = zeros(Float64,T)

smc²(ucsv_smc²,y)

for t in 2:T
    ucsv_xqs[t-1],ucsv_cqs[t-1],ucsv_vars[t-1] = get_quantiles_ucsv(ucsv_smc²,y[t-1])
    smc²!(ucsv_smc²,y,t)
end

ucsv_xqs[T],ucsv_cqs[T],ucsv_vars[T] = get_quantiles_ucsv(ucsv_smc²,y[T])


## Plot Results ###############################################################

ucsv_plt1 = scatter(
    pce_data.value,
    xformatter = x -> pce_data.date[Int(x+1)],
    framestyle = :box,
    color = :black,
    label = "observed data"
)

plot!(
    hcat(ucsv_xqs...)[1,:],
    fillrange = hcat(ucsv_xqs...)[3,:],
    fillcolor = :grey,
    fillalpha = 0.35,
    linealpha = 0.0,
    label = false
)

plot!(
    hcat(ucsv_xqs...)[2,:],
    color = :red,
    ylims = (-5,15),
    label = "filtered trend (UCSV)",
    title = "quarterly PCE inflation rate"
)

ucsv_plt2 = plot(
    hcat(ucsv_cqs...)[1,:],
    xformatter = x -> pce_data.date[Int(x+1)],
    fillrange = hcat(ucsv_cqs...)[3,:],
    fillcolor = :grey,
    fillalpha = 0.5,
    linealpha = 0.0,
    label = false
)

plot!(
    hcat(ucsv_cqs...)[2,:],
    color = :red,
    framestyle = :box,
    ylims = (-5,5),
    label = "filtered cycle (UCSV)",
    title = "quarterly PCE inflation rate"
)

savefig(ucsv_plt1,"visuals/pce_inflation_trend_1960-2020_ucsv.svg")
savefig(ucsv_plt2,"visuals/pce_inflation_cycle_1960-2020_ucsv.svg")


## Run Bootstrap Filter #######################################################

pred_ucsv_θ = expected_parameters(ucsv_smc²)
pred_ucsv_mod = ucsv_mod(pred_ucsv_θ)

function get_latent_states_ucsv(
        rng::AbstractRNG,
        N::Int64,
        y::Vector{Float64},
        model::StateSpaceModel,
        proposal=nothing
    )
    xq = fill(zeros(Float64,3),length(y))
    cq = fill(zeros(Float64,3),length(y))
    vars = zeros(length(y))

    x,w,_ = particle_filter(rng,N,y[1],model,proposal)
    x_cloud = reduce(hcat,x)[1,:]

    xq[1] = quantile(x_cloud,[0.25,0.5,0.75])
    cq[1] = quantile(y[1].-x_cloud,[0.25,0.5,0.75])

    for t in 2:length(y)
        _,w,_ = particle_filter!(rng,x,w,y[t],model,proposal)
        x_cloud = reduce(hcat,x)[1,:]
        xq[t] = quantile(x_cloud,[0.25,0.5,0.75])
        cq[t] = quantile(y[t].-x_cloud,[0.25,0.5,0.75])
        vars[t] = var(x_cloud)
    end

    return xq,cq,vars
end

ucsv_pred_xqs,ucsv_pred_cqs,ucsv_pred_vars = get_latent_states_ucsv(Random.GLOBAL_RNG,1024,y,pred_ucsv_mod)

pred_ucsv_plt1 = scatter(
    pce_data.value,
    framestyle = :box,
    color = :black,
    label = "observed data"
)

plot!(
    hcat(ucsv_pred_xqs...)[1,:],
    xformatter = x -> pce_data.date[Int(x+1)],
    fillrange = hcat(ucsv_pred_xqs...)[3,:],
    fillcolor = :grey,
    fillalpha = 0.5,
    linealpha = 0.0,
    label = false
)

plot!(
    hcat(ucsv_pred_xqs...)[2,:],
    linecolor = :red,
    label = "filtered trend (UCSV)",
    title = "quarterly PCE inflation rate (given θ)"
)

pred_ucsv_plt2 = plot(
    hcat(ucsv_pred_cqs...)[1,:],
    xformatter = x -> pce_data.date[Int(x+1)],
    fillrange = hcat(ucsv_pred_cqs...)[3,:],
    fillcolor = :grey,
    fillalpha = 0.5,
    linealpha = 0.0,
    label = false
)

plot!(
    hcat(ucsv_pred_cqs...)[2,:],
    color = :red,
    framestyle = :box,
    ylims = (-5,5),
    label = "filtered cycle (UCSV)",
    title = "quarterly PCE inflation rate (given θ)"
)

savefig(pred_ucsv_plt1,"pce_inflation_trend_1960-2020_ucsv_post.pdf")
savefig(pred_ucsv_plt2,"pce_inflation_cycle_1960-2020_ucsv_post.pdf")


var_plt = plot(
    log.(uc_vars) .- log.(uc_pred_vars),
    xformatter = x -> pce_data.date[Int(x+1)],
    label = "log variance ratio (UC)",
    title = "ratio of var(P(x,θ|y)) to var(P(x|y,θ))",
    framestyle = :box,
    color = :blue
)


plot!(
    log.(ucsv_vars) .- log.(ucsv_pred_vars),
    label = "log variance ratio (UCSV)",
    #title = "ratio of var(P(x,θ|y)) to var(P(x|y,θ))",
    framestyle = :box,
    color = :red,
    ylims = (-2,10)
)

savefig(var_plt,"log_variance_ratio_inflation.pdf")
