# The operator hierarchy
abstract type AbstractOperator end
abstract type Operator <: AbstractOperator end
abstract type SuperOperator <: AbstractOperator end

# The general interface to operator-state applications
(op::AbstractOperator)(state::SparseState) = apply(op, state)

apply(op::AbstractOperator, state::SparseState; kwargs...) = sort!(apply!(op, copy(state); kwargs...))

function Base.show(io::IO, op::O) where {O<:AbstractOperator}
    indices = support(op)
    if length(indices) == 1
        print(io, "$(O)($(join(only(indices), ", ")))")
    else
        if eltype(indices) == Tuple{Int}
            print(io, "$(O)([$(join((only(i) for i in indices), ", "))])")
        else
            print(io, "$(O)([$(join(indices, ", "))])")
        end
    end
    return nothing
end

# Overload three-argument dot product
function LinearAlgebra.dot(first_state::SparseState{K}, op::AbstractOperator, second_state::SparseState{K}) where {K}
    return dot(first_state, apply(op, second_state))
end

function support end

macro operator(id, N, pairs...)
    # Parse identifier
    if id isa Expr
        if id.head == :(<:)
            Name, Parent = id.args
        else
            error("Expected name and parent to be of the form `Name::Parent`")
        end
    else
        Name = id
        Parent = Operator
    end

    # params = []
    # for expr in pairs
    #     if expr.head == :(=)
    #         # Default argument
    #         type, value = expr.args
    #         push!(params, type)
    #     elseif expr.head == :(::)
    #         # No default argument
    #         push!(params, expr)
    #     else
    #         error("Expected optional arguments of the form `field::Type [=default]`")
    #     end
    # end

    return quote
        struct $Name <: $Parent
            # $(params...)
            support::Vector{NTuple{$N,Int}}

            $Name(support::AbstractVector{NTuple{$N,Int}}) = new(support)
        end

        $(esc(Name))(inds::AbstractArray) = $(esc(Name))(vec(map(NTuple{$N,Int}, inds)))
        $(esc(Name))(inds::Vararg{<:AbstractArray{Int},$N}) = $(esc(Name))(map(tuple, inds...))
        $(esc(Name))(inds::Vararg{Int,$N}) = $(esc(Name))([inds])
        $(esc(Name))(iterable) = $(esc(Name))(vec(collect(iterable)))

        SparseStates.support(op::$(esc(Name))) = op.support
    end
end
