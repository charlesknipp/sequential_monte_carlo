module SequentialMonteCarlo

using Distributions,LinearAlgebra,Statistics,Printf,Random,StatsBase
import Statistics: quantile

# state space object definition
include("state_space_models.jl")

# for generalized state space models
include("particle_filter.jl")
include("smc_samplers.jl")

# for linear models
include("kalman_filter.jl")
include("ibis.jl")

# in and out of sample forecasting
include("forecast.jl")

# custom plotting functions (WIP)
using PGFPlotsX,LaTeXStrings,DataFrames,ColorBrewer
include("plotting_utils.jl")

# fast multivariate distributions
using PDMats,StaticArrays
include("utilities.jl")

end
