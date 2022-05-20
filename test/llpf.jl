using LowLevelParticleFilters, LinearAlgebra, StaticArrays, Distributions

# Define problem

nx = 2   # Dimension of state
nu = 2   # Dimension of input
ny = 2   # Dimension of measurements
N = 500 # Number of particles

const dg = MvNormal(Diagonal(ones(ny)))          # Measurement noise Distribution
const df = MvNormal(Diagonal(ones(nx)))          # Dynamics noise Distribution
const dx0 = MvNormal(randn(nx),2.0^2*I)   # Initial state Distribution

# Define random linear state-space system
Tr = randn(nx,nx)
const A_lg = SMatrix{nx,nx}(Tr*diagm(0=>LinRange(0.5,0.95,nx))/Tr)
const B_lg = @SMatrix randn(nx,nu)
const C_lg = @SMatrix randn(ny,nx)

# The following two functions are required by the filter
dynamics(x,u,p,t) = A_lg*x .+ B_lg*u
measurement(x,u,p,t) = C_lg*x
#vecvec_to_mat(x) = copy(reduce(hcat, x)') # Helper function

eye(n) = Matrix{Float64}(I,n,n)
kf = KalmanFilter(A_lg, B_lg, C_lg, 0, eye(nx), eye(ny), MvNormal(Diagonal([1.,1.])))

pf = ParticleFilter(N, dynamics, measurement, df, dg, dx0)
xs,u,y = simulate(pf,200,df) # We can simulate the model that the pf represents

loglik(pf,u,y)