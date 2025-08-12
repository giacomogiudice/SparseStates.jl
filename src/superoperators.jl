# Cache all combinations up to support of length 3
const PAULI_COMBINATIONS = ntuple(l -> collect(pauli_combinations(l)), 3)

default_callback(outcomes, state::SparseState) = nothing

@super_operator Reset{F} 1 callback::F = default_callback
@super_operator Measure{F} 1 callback::F = default_callback

function apply!(channel::Measure, state::SparseState{K,V}) where {K,V}
    (; callback) = channel
    (; table, masks) = state
    @inbounds outcomes = map(support(channel)) do (i,)
        p = expectation(state, i)
        outcome = rand() < p
        m = state.masks[i]
        c = outcome ? 1 / √p : 1 / √(1 - p)
        n = firstindex(table)
        while n <= lastindex(table)
            s, v = table[n]
            if iszero(s & m) ⊻ outcome
                table[n] = s => c * v
                n += 1
            else
                popat!(table, n)
            end
        end
        return outcome
    end
    callback(outcomes, state)
    return state
end

function apply!(channel::Reset, state::SparseState{K,V}) where {K,V}
    (; callback) = channel
    (; table, masks) = state
    @inbounds outcomes = map(support(channel)) do (i,)
        p = expectation(state, i)
        outcome = rand() < p
        m = state.masks[i]
        c = outcome ? 1 / √p : 1 / √(1 - p)
        n = firstindex(table)
        while n <= lastindex(table)
            s, v = table[n]
            if iszero(s & m) ⊻ outcome
                table[n] = s & ~m => c * v
                n += 1
            else
                popat!(table, n)
            end
        end
        return outcome
    end
    setsorted!(state, false)
    callback(outcomes, state)
    return state
end

struct MeasureOperator{O<:AbstractOperator,F} <: SuperOperator
    ops::Vector{O}
    callback::F
end

function MeasureOperator(ops::AbstractArray{O}; callback::F=default_callback) where {O<:AbstractOperator,F}
    return MeasureOperator(vec(collect(ops)), callback)
end

MeasureOperator(ops::AbstractOperator...; kwargs...) = MeasureOperator([ops...]; kwargs...)
MeasureOperator(iterable; kwargs...) = MeasureOperator(vec(collect(iterable)); kwargs...)

Base.parent(channel::MeasureOperator) = channel.ops
support(channel::MeasureOperator) = mapreduce(support, vcat, parent(channel); init=Tuple{Vararg{Int}}[])

function Base.show(io::IO, channel::MeasureOperator)
    (; ops, callback) = channel
    return show_operator(io, :MeasureOperator, ops; callback)
end

function apply!(channel::MeasureOperator, state::SparseState)
    (; ops, callback) = channel
    sort!(state)
    @inbounds outcomes = map(ops) do op
        new_state = apply(op, state)
        # Probability of having outcome `false` (+1 eigenstate) is `(1 + real(⟨U⟩) / 2`
        p = (1 - real(dot(state, new_state)) / (norm(state) * norm(new_state))) / 2
        outcome = rand() < p
        c = outcome ? 1 / √p : 1 / √(1 - p)
        # Output state is `(state ± new_state) / 2`, merge states to perform addition
        outcome && rmul!(new_state, -1)
        sorted_merge!(state, new_state)
        rmul!(state, c / 2)
        return outcome
    end
    callback(outcomes, state)
    return state
end

struct DepolarizingChannel{N,F<:Function} <: SuperOperator
    support::Vector{NTuple{N,Int}}
    p::Float64
    callback::F

    function DepolarizingChannel{N}(
        support::AbstractVector{NTuple{N,Int}}; p::Number=0, callback::F=default_callback
    ) where {N,F<:Function}
        return new{N,F}(support, p, callback)
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
    @inbounds outcomes = map(support(channel)) do inds
        outcome = rand() < p
        if outcome
            ops = rand(@view PAULI_COMBINATIONS[N][(begin + 1):end])
            circuit = mapreduce((O, i) -> O(i), *, ops, inds)
            apply!(circuit, state)
        end
        return outcome
    end
    callback(outcomes, state)
    return state
end
