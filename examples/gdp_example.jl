include(joinpath(pwd(),"src/sequential_monte_carlo.jl"))

using .SequentialMonteCarlo
using Random,Distributions,LinearAlgebra
#using StaticArrays
using Plots
using FredData

# get quarterly GDP using the FRED API (ecae2fc8d6c684847525a828ae7a3ab8)
gdp_series = get_data(Fred(),"GDPC1").data
gdp_data   = gdp_series.value

# define a state space model for GDP trend decomposition