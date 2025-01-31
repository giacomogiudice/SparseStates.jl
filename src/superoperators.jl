using Base: Callable

# Cache all combinations up to support of length 3
const PAULI_COMBINATIONS = ntuple(l -> collect(pauli_combinations(l)), 3)

@super_operator Reset 1
@super_operator Measure{C<:Callable} 1 callback::C = Returns(nothing)

function apply!(channel::Measure, state::SparseState{K,V}) where {K,V}
    (; callback) = channel
    (; table, masks) = state
    outcomes = Bool[]
    sizehint!(outcomes, length(support(channel)))
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
    callback(outcomes)
    return state
end

function apply!(channel::Reset, state::SparseState{K,V}) where {K,V}
    (; table, masks) = state
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
    end
    return state
end

struct MeasureOperator{O<:AbstractOperator,C<:Callable} <: SuperOperator
    op::O
    callback::C
end

function MeasureOperator(op::O; callback::C=Returns(nothing)) where {O<:AbstractOperator,C<:Callable}
    return MeasureOperator(op, callback)
end

support(channel::MeasureOperator) = support(channel.op)

function apply!(channel::MeasureOperator, state::SparseState)
    (; op, callback) = channel
    new_state = apply(op, state)
    # Probability of having outcome `false` (+1 eigenstate) is `(1 + real(⟨U⟩) / 2`
    p = (1 - real(dot(state, new_state))) / 2
    outcome = rand() < p
    # Output state is `(state ± new_state) / 2`
    if outcome
        new_state *= -1
    end
    # Merge states to perform addition
    sorted_merge!(state, new_state)
    rmul!(state, 1 / 2)
    callback([outcome])
    return state
end

struct DepolarizingChannel{N,C<:Callable} <: SuperOperator
    support::Vector{NTuple{N,Int}}
    p::Float64
    callback::C

    function DepolarizingChannel{N}(
        support::AbstractVector{NTuple{N,Int}}; p::Number=0, callback::C=Returns(nothing)
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
    outcomes = Bool[]
    sizehint!(outcomes, length(support(channel)))
    for inds in support(channel)
        outcome = rand() < p
        if outcome
            ops = rand(@view PAULI_COMBINATIONS[N][(begin + 1):end])
            circuit = mapreduce((O, i) -> O(i), *, ops, inds)
            apply!(circuit, state)
        end
        push!(outcomes, outcome)
    end
    callback(outcomes)
    return state
end
