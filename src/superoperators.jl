using Base: Callable

# Cache all combinations up to support of length 3
const PAULI_COMBINATIONS = ntuple(l -> collect(pauli_combinations(l)), 3)

default_callback(outcomes, state::SparseState) = nothing

@super_operator Reset{C<:Callable} 1 callback::C = default_callback
@super_operator Measure{C<:Callable} 1 callback::C = default_callback

function apply!(channel::Measure, state::SparseState{K,V}) where {K,V}
    (; callback) = channel
    (; table, masks) = state
    outcomes = sizehint!(Bool[], length(support(channel)))
    @inbounds for (i,) in support(channel)
        p = expectation(state, i)
        outcome = rand() < p
        m = state.masks[i]
        n = outcome ? √p : √(1 - p)
        i = firstindex(table)
        while i <= lastindex(table)
            s, v = table[i]
            if outcome ? (s & m == m) : iszero(s & m)
                table[i] = s => v / n
                i += 1
            else
                popat!(table, i)
            end
        end
        push!(outcomes, outcome)
    end
    callback(outcomes, state)
    return state
end

function apply!(channel::Reset, state::SparseState{K,V}) where {K,V}
    (; callback) = channel
    (; table, masks) = state
    outcomes = sizehint!(Bool[], length(support(channel)))
    @inbounds for (i,) in support(channel)
        p = expectation(state, i)
        outcome = rand() < p
        m = state.masks[i]
        n = outcome ? √p : √(1 - p)
        i = firstindex(table)
        while i <= lastindex(table)
            s, v = table[i]
            if outcome ? (s & m == m) : iszero(s & m)
                table[i] = s & ~m => v / n
                i += 1
            else
                popat!(table, i)
            end
        end
        push!(outcomes, outcome)
    end
    callback(outcomes, state)
    return state
end

struct MeasureOperator{O<:AbstractOperator,C<:Callable} <: SuperOperator
    ops::Vector{O}
    callback::C
end

function MeasureOperator(ops::AbstractArray{O}; callback::C=default_callback) where {O<:AbstractOperator,C<:Callable}
    return MeasureOperator(vec(collect(ops)), callback)
end

MeasureOperator(ops::AbstractOperator...; kwargs...) = MeasureOperator([ops...]; kwargs...)
MeasureOperator(iterable; kwargs...) = MeasureOperator(vec(collect(iterable)); kwargs...)

Base.parent(channel::MeasureOperator) = channel.ops
support(channel::MeasureOperator) = mapreduce(support, vcat, parent(channel); init=Tuple{Vararg{Int}}[])

function Base.show(io::IO, channel::MeasureOperator)
    (; ops, callback) = channel

    print(io, "MeasureOperator", "(")
    if length(ops) == 1
        print(io, only(ops))
    else
        print(io, "[", join(ops, ", "), "]")
    end
    print(io, "; ", "callback=$(callback)")
    print(io, ")")

    return nothing
end

function apply!(channel::MeasureOperator, state::SparseState)
    (; ops, callback) = channel
    sort!(state)
    outcomes = sizehint!(Bool[], length(ops))
    for op in ops
        new_state = apply(op, state)
        # Probability of having outcome `false` (+1 eigenstate) is `(1 + real(⟨U⟩) / 2`
        p = (1 - real(dot(state, new_state))) / 2
        outcome = rand() < p
        n = outcome ? √p : √(1 - p)
        # Output state is `(state ± new_state) / 2`
        if outcome
            rmul!(new_state, -1)
        end
        # Merge states to perform addition
        sorted_merge!(state, new_state)
        rmul!(state, 1 / 2n)
        push!(outcomes, outcome)
    end
    callback(outcomes, state)
    return state
end

struct DepolarizingChannel{N,C<:Callable} <: SuperOperator
    support::Vector{NTuple{N,Int}}
    p::Float64
    callback::C

    function DepolarizingChannel{N}(
        support::AbstractVector{NTuple{N,Int}}; p::Number=0, callback::C=default_callback
    ) where {N,C<:Callable}
        return new{N,C}(support, p, callback)
    end
end

function DepolarizingChannel{N}(inds::Vararg{AbstractArray{Int},N}; kwargs...) where {N}
    return DepolarizingChannel{N}(vec(map(tuple, inds...)); kwargs...)
end
function DepolarizingChannel(inds::AbstractArray{NTuple{N,Int}}; kwargs...) where {N}
    return DepolarizingChannel{N}(vec(inds); kwargs...)
end
function DepolarizingChannel(inds::Vararg{AbstractArray{Int},N}; kwargs...) where {N}
    return DepolarizingChannel{N}(inds...; kwargs...)
end
function DepolarizingChannel(inds::Vararg{Int,N}; kwargs...) where {N}
    return DepolarizingChannel{N}([inds]; kwargs...)
end

DepolarizingChannel{N}(; kwargs...) where {N} = (inds...) -> DepolarizingChannel{N}(inds...; kwargs...)

support(channel::DepolarizingChannel) = channel.support

function apply!(channel::DepolarizingChannel{N}, state::SparseState) where {N}
    (; p, callback) = channel
    outcomes = sizehint!(Bool[], length(support(channel)))
    for inds in support(channel)
        outcome = rand() < p
        if outcome
            ops = rand(@view PAULI_COMBINATIONS[N][(begin + 1):end])
            circuit = mapreduce((O, i) -> O(i), *, ops, inds)
            apply!(circuit, state)
        end
        push!(outcomes, outcome)
    end
    callback(outcomes, state)
    return state
end
