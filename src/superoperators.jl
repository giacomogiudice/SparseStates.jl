# Cache all combinations up to support of length 3
const PAULI_COMBINATIONS = ntuple(l -> collect(pauli_combinations(l)), 3)

struct DepolarizingChannel{N} <: SuperOperator
    p::Float64
    support::Vector{NTuple{N,Int}}

    DepolarizingChannel{N}(p::Number, obs::AbstractVector{NTuple{N,Int}}) where {N} = new{N}(p, obs)
end

function DepolarizingChannel{N}(p::Number, inds::Vararg{AbstractArray{Int},N}) where {N}
    return DepolarizingChannel{N}(p, vec(map(NTuple{N,Int}, inds...)))
end

DepolarizingChannel{N}(p::Number, inds) where {N} = DepolarizingChannel{N}(p, collect(inds))
DepolarizingChannel(p::Number, inds::AbstractArray{NTuple{N,Int}}) where {N} = DepolarizingChannel{N}(p, vec(inds))
DepolarizingChannel(p::Int, inds::Vararg{AbstractArray{Int},N}) where {N} = DepolarizingChannel{N}(p, inds...)
function DepolarizingChannel(inds::Vararg{<:Union{Int,AbstractArray{Int}},N}; p::Number) where {N}
    return DepolarizingChannel{N}(p, inds...)
end
DepolarizingChannel{N}(p::Number) where {N} = (inds...) -> DepolarizingChannel{N}(p, inds...)

support(channel::DepolarizingChannel) = channel.support

function apply!(channel::DepolarizingChannel{N}, state::SparseState) where {N}
    (; p) = channel
    for inds in support(channel)
        if rand() < p
            ops = rand(@view PAULI_COMBINATIONS[N][(begin + 1):end])
            circuit = mapreduce((O, i) -> O(i), *, ops, inds)
            apply!(circuit, state)
        end
    end
    return state
end

@operator(Measure <: SuperOperator, 1)
@operator(Reset <: SuperOperator, 1)

function apply!(channel::Measure, state::SparseState{K,V}) where {K,V}
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
                table[i] = s => v / n
                i += 1
            else
                popat!(table, i)
            end
        end
    end
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
