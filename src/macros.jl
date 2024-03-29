
# define the abstract type
abstract type StateSpaceModel end

"""
@model (opt) function model_name(default_parameterizations...)

This macro performs all of the overhead to construct custom state space models
```julia
@model function ucsv(x0=0.0)
    # initial density
    @init begin
        x  = Normal(x0,exp(0.5*σx0))
        σx = Normal(σx0,γ)
        σy = Normal(σy0,γ)
    end

    # transition density
    σx = Normal(@prev(σx),γ)
    σy = Normal(@prev(σy),γ)
	x  = Normal(@prev(x),exp(0.5*@prev(σx)))

    # observation density
    @observe y = Normal(x,exp(0.5*σy))
end
```
"""
macro model(args...)
    model_definition = splitdef(args[end])
    #println(MacroTools.prettify(build_model(model_definition)))

    if args[1] == :linear
        # for linear models constructed to perform kalman filtering
        return build_linear_model(model_definition)
    else
        # for non-linear models intended for particle filtering
        return build_model(model_definition)
    end
end

# construct the framework
function build_model(expr)
    model_build = expr[:body]
    model_name  = expr[:name]

    # infer the parameters given the expression (not finished)
    all_parameters = extract_parameters(model_build)
    default_params = set_default_parameters(expr[:args])
    parameters = setdiff(all_parameters,default_params[1])

    # store assignments as dictionaries
    new_build,init_vars = initialize_model_vars(model_build)
    new_build,obs_vars  = initialize_observations(new_build)

    # TODO: use final_build to make intermediary calculations
    final_build,latent_vars,trans_vars = define_transition(new_build)

    # use latent_states to define the order of distributions
    init_ops  = Vector{Expr}()
    trans_ops = Vector{Expr}()
    for var in latent_vars
        push!(init_ops,get(init_vars,var,:(nothing)))
        push!(trans_ops,get(trans_vars,var,:(nothing)))
    end

    model_object = quote
        struct $(model_name) <: StateSpaceModel end
    end

    @gensym latent_state
    unpacked_params   = Expr(:tuple,collect(parameters)...)
    unpacked_states   = Expr(:tuple,collect(latent_vars)...)
    unpacked_defaults = Expr(:tuple,collect(default_params[1])...)

    # define the returns accounting for product_dist construction
    function define_return(ops)
        if length(ops) == 1
            return ops[1]
        else
            vec_ops = Expr(:tuple,ops...)
            return :(TupleProduct($(vec_ops)))
        end
    end

    init_return  = define_return(init_ops)
    trans_return = define_return(trans_ops)
    obs_return   = define_return(collect(values(obs_vars)))

    # this assumes that each state variable is univariate...not great
    if length(init_ops) == 1
        prealloc_return = :(zeros(Float64,N))
        XT = Float64
    else
        dim_states = length(init_ops)
        prealloc_return = :(fill(zeros(SVector{$(dim_states)}),N))
        XT = SVector{dim_states,Float64}
    end

    # NOTE: there isn't any type inference so keep it abstract
    init_func = esc(quote
        function initial_dist(::Type{$(model_name)},θ)
            $(unpacked_defaults) = $(default_params[2])
            $(unpacked_params) = θ
            return $(init_return)
        end
    end)

    trans_func = esc(quote
        function transition(::Type{$(model_name)},$(latent_state)::$(XT),θ)
            $(unpacked_defaults) = $(default_params[2])
            $(unpacked_params) = θ
            $(unpacked_states) = $(latent_state)
            $(clean_expr(final_build))
            return $(trans_return)
        end
    end)

    obs_func = esc(quote
        function observation(::Type{$(model_name)},$(latent_state)::$(XT),θ)
            $(unpacked_defaults) = $(default_params[2])
            $(unpacked_params) = θ
            $(unpacked_states) = $(latent_state)
            return $(obs_return)
        end
    end)

    misc_funcs = esc(quote
        # preallocate a cloud of state particles using this constructor
        function preallocate(::Type{$(model_name)},N::Int64)
            return $(prealloc_return)
        end

        # return fetch as either a tuple or a set, not sure yet
        get_parameters(::Type{$(model_name)}) = $(tuple(parameters...))
        get_states(::Type{$(model_name)}) = $(tuple(latent_vars...))
    end)

    return quote
        $(model_object)
        $(trans_func)
        $(obs_func)
        $(init_func)
        $(misc_funcs)
    end
end

# for now this is yet to be implemented...
function build_linear_model(expr)
    @warn "Linear models are not yet implemented"
    return build_model(expr)
end

# clean a block by removing lines
function clean_expr(expr)
    not_nothing = Vector{Expr}()
    for arg in expr.args
        if arg isa Expr
            push!(not_nothing,arg)
        end
    end

    return not_nothing
end

# redefine expression walking across variable calls
sym_walk(x,inner,outer) = outer(x)
sym_walk(x::Expr,inner,outer) = outer(Expr(x.head,map(inner,x.args[2:end])...))
sym_postwalk(f,x) = sym_walk(x,x->sym_postwalk(f,x),f)

# NOTE: not sure I want to infer these or use @parameters
function extract_parameters(expr::Expr)
    vars = Set{Symbol}()
    LHS  = Set{Symbol}()

    # get all called variables
    sym_postwalk(expr) do ex
        if ex isa Symbol
            push!(vars,ex)
            return ex
        else
            return ex
        end
    end

    # get all variables which get assigned during model construction
    postwalk(expr) do ex
        if @capture(ex,var_ = val_)
            push!(LHS,var)
            return ex
        else
            return ex
        end
    end

    # the difference should be the model parameters
    return setdiff(vars,LHS)
end

# for parameters specified in the function arg...
function set_default_parameters(expr::AbstractVector)
    params = Set{Symbol}()
    values = Vector{Real}()
    for exp in expr
        push!(params,exp.args[1])
        push!(values,eval(exp.args[2]))
    end
    return params,values
end

# NOTE: this does not work when intermediary calculations are made
function initialize_model_vars(expr::Expr)
    # two scenarios here: we initialize in one line or in a block
    init_vars = Dict{Symbol,Expr}()
    for i in 1:length(expr.args)
        if @capture(expr.args[i],@init begin defs__ end)
            postwalk(expr.args[i]) do sub_ex
                if @capture(sub_ex,var_ = val_)
                    init_vars[var] = val
                    return nothing
                else
                    return sub_ex
                end
            end
            expr.args[i] = nothing
        elseif @capture(expr.args[i],@init var_ = val_)
            init_vars[var] = val
        end
    end
    return expr,init_vars
end

function initialize_observations(expr::Expr)
    # again two scenarios: we initialize in one line or in a block
    obs_vars = Dict{Symbol,Expr}()
    for i in 1:length(expr.args)
        if @capture(expr.args[i],@observe begin defs__ end)
            postwalk(expr.args[i]) do sub_ex
                if @capture(sub_ex,var_ = val_)
                    obs_vars[var] = val
                    return nothing
                else
                    return sub_ex
                end
            end
            expr.args[i] = nothing
        elseif @capture(expr.args[i],@observe var_ = val_)
            obs_vars[var] = val
        end
    end
    return expr,obs_vars
end

function define_transition(expr::Expr)
    # treat the prev calls and define latent state vector
    latent_states = Set{Symbol}()
    clean_expr = postwalk(expr) do ex
        if @capture(ex,@prev(var_))
            push!(latent_states,var)
            return var
        else
            return ex
        end
    end

    # define the transitions and leave intermediary calculations
    trans_vars = Dict{Symbol,Expr}()
    new_expr = postwalk(clean_expr) do ex
        if @capture(ex,var_ = val_) && var in latent_states
            trans_vars[var] = val
            return nothing
        else
            return ex
        end
    end

    # check to see if prev calls are transitioned
    for state in latent_states
        if !haskey(trans_vars,state)
            error("bad @prev call")
        end
    end

    # TODO: record intermediary variable names
    return new_expr,collect(latent_states),trans_vars
end
