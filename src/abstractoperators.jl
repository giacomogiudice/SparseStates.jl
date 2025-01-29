# The operator hierarchy
abstract type AbstractOperator end
abstract type Operator <: AbstractOperator end
abstract type SuperOperator <: AbstractOperator end

# The general interface to operator-state applications
(op::AbstractOperator)(state::SparseState) = apply(op, state)

apply(op::AbstractOperator, state::SparseState; kwargs...) = sort!(apply!(op, copy(state); kwargs...))

function Base.show(io::IO, op::O) where {O<:AbstractOperator}
    indices = support(op)
    parameters = [name => getfield(op, name) for name in fieldnames(O) if name ≠ :support]
    
    print(io, nameof(O), "(")
    if length(indices) == 1
        print(io, join(only(indices), ", "))
    else
        if eltype(indices) == Tuple{Int}
            print(io, "[", join((only(i) for i in indices), ", "), "]")
        else
            print(io, "[", join(indices, ", "), "]")
        end
    end
    if !isempty(parameters)
        print(io, "; ", join(("$(name)=$(value)" for (name, value) in parameters), ", "))
    end
    print(io, ")")

    return nothing
end

# Overload three-argument dot product
function LinearAlgebra.dot(first_state::SparseState{K}, op::AbstractOperator, second_state::SparseState{K}) where {K}
    return dot(first_state, apply(op, second_state))
end

function support end

# Define `@operator` and `@super_operator` macros
function parse_field(expr::Union{Symbol,Expr})
    if expr isa Symbol
        key = expr
        type = Any
        value = ()
    elseif Meta.isexpr(expr, :(::)) || Meta.isexpr(expr, :(<:))
        key, type = expr.args
        value = ()
    elseif Meta.isexpr(expr, :(=))
        sub_expr, value = expr.args
        key, type, _ = parse_field(sub_expr)
    else
        throw(ArgumentError("Expected optional arguments of the form `field[::type] [=default], got $(expr)`"))
    end
    return key, type, value
end

function generic_operator(parent, identifier, N, fields...)
    if Meta.isexpr(identifier, :curly)
        struct_name, parametric_types... = identifier.args
    else
        struct_name = identifier
        parametric_types = []
    end
    parsed_fields = map(parse_field, fields)
    
    names = [key for (key, _, _) in parsed_fields]
    types = [first(parse_field(expr)) for expr in parametric_types]
    names_with_types = [Expr(:(::), key, type) for (key, type, _) in parsed_fields]
    names_with_types_and_values = [Expr(:kw, Expr(:(::), key, type), value) for (key, type, value) in parsed_fields]
    return quote
        struct $identifier <: $parent
            support::Vector{NTuple{$N,Int}}
            $(names_with_types...)

            function $struct_name(support::AbstractVector{NTuple{$N,Int}}; $(names_with_types_and_values...)) where {$(parametric_types...)}
                return new{$(types...)}(support, $(names...))
            end
        end

        $(esc(struct_name))(inds::AbstractArray; kwargs...) = $(esc(struct_name))(vec(map(NTuple{$N,Int}, inds)); kwargs...)
        $(esc(struct_name))(inds::AbstractArray{Int}...; kwargs...) = $(esc(struct_name))(map(tuple, inds...); kwargs...)
        $(esc(struct_name))(inds::Int...; kwargs...) = $(esc(struct_name))([inds]; kwargs...)
        $(esc(struct_name))(iterable; kwargs...) = $(esc(struct_name))(vec(collect(iterable)); kwargs...)

        SparseStates.support(op::$(esc(struct_name))) = op.support
    end
end

macro operator(identifier, N, parameters...)
    return generic_operator(Operator, identifier, N, parameters...)
end

macro super_operator(identifier, N, parameters...)
    return generic_operator(SuperOperator, identifier, N, parameters...)
end
