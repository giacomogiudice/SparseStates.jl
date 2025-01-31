using Base: Callable

const DEFAULT_KEYTYPE = UInt64
const DEFAULT_ELTYPE = Complex{Float64}

parse_key(key, masks::AbstractVector{K}) where {K} = convert(K, key)

function parse_key(key::AbstractString, masks::AbstractVector{K}) where {K}
    @boundscheck length(key) == length(masks) ||
        throw(ArgumentError("Key string length does not correspond to the number of masks"))

    return mapreduce(|, key, masks) do c, m
        if c == '0'
            return zero(K)
        elseif c == '1'
            return m
        else
            throw(ArgumentError("key $(key) is expected to only have 0s and 1s"))
        end
    end
end

struct SparseState{K,V} <: AbstractDict{K,V}
    table::Vector{Pair{K,V}}
    masks::Vector{K}
end

function SparseState{K,V}(pairs, qubits::Int) where {K,V}
    qubits > 8 * sizeof(K) &&
        throw(ArgumentError("The number of qubits $qubits exceeds the number of bits in the keytype $K"))
    masks = map(n -> one(K) << n, 0:(qubits - 1))
    iter = Iterators.map(pairs) do (key, val)
        return parse_key(key, masks) => val
    end
    return SparseState{K,V}(sort!(vec(collect(iter)); by=first), masks)
end

SparseState{K}(pairs, qubits::Int) where {K} = SparseState{K,DEFAULT_ELTYPE}(pairs, qubits)
SparseState(pairs, qubits::Int) = SparseState{DEFAULT_KEYTYPE}(pairs, qubits)
SparseState(pairs::AbstractVector{Pair{K,V}}, qubits::Int) where {K<:Integer,V} = SparseState{K,V}(pairs, qubits)

SparseState{K,V}(pair::Pair, qubits::Int) where {K,V} = SparseState{K,V}([pair], qubits)
SparseState(pair::Pair, qubits::Int) = SparseState([pair], qubits)

SparseState{K,V}(qubits::Int) where {K,V} = SparseState{K,V}([zero(K) => one(V)], qubits)
SparseState{K}(qubits::Int) where {K} = SparseState{K,DEFAULT_ELTYPE}(qubits)
SparseState(qubits::Int) = SparseState{DEFAULT_KEYTYPE}(qubits)

table(state::SparseState) = state.table
num_qubits(state::SparseState) = length(state.masks)

Base.length(state::SparseState) = length(table(state))
Base.empty(state::SparseState, K, V) = SparseState{K,V}([], state.masks)
Base.copy(state::SparseState) = SparseState(copy(state.table), copy(state.masks))
Base.sizehint!(state::SparseState, n::Integer) = sizehint!(table(state), n)
Base.iterate(state::SparseState, args...) = iterate(table(state), args...)
Base.pairs(state::SparseState) = (pair for pair in table(state))

function Base.sort!(state::SparseState)
    sort!(table(state); by=first)
    return state
end

function Base.haskey(state::SparseState, key)
    (; table, masks) = state
    key = parse_key(key, masks)
    i = searchsortedfirst(table, key; by=first)
    return i ≤ length(table) && first(table[i]) == key
end

function Base.get(state::SparseState, key, default)
    (; table, masks) = state
    key = parse_key(key, masks)
    i = searchsortedfirst(table, key; by=first)
    if i ≤ length(table)
        pair = table[i]
        if first(pair) == key
            return last(pair)
        end
    end
    return default
end

function Base.get!(default::Callable, state::SparseState, key)
    (; table, masks) = state
    key = parse_key(key, masks)
    i = searchsortedfirst(table, key; by=first)
    if i ≤ length(table)
        pair = table[i]
        if first(pair) == key
            return last(pair)
        end
    end
    val = convert!(valtype(state), default())
    insert!(table, i, val)
    return val
end

function Base.getindex(state::SparseState, key::AbstractString)
    (; masks) = state
    return getindex(state, parse_key(key, masks))
end

function Base.setindex!(state::SparseState, key, value)
    key = convert_to_keytype(state, key)
    value = convert(valtype(state), value)
    table = table(state)
    i = searchsortedfirst(table, key; by=first)
    if i ≤ length(table)
        pair = table[i]
        if first(pair) == key
            table[i] = key => value
        end
    end
    insert!(table, i, key => value)
    return value
end

function Base.show(io::IO, state::SparseState{K,V}) where {K,V}
    return print(io, "SparseState{$K,$V} with $(num_qubits(state)) qubits")
end

function Base.show(io::IO, ::MIME"text/plain", state::SparseState{K,V}) where {K,V}
    qubits = num_qubits(state)
    println(io, "SparseState{$K,$V} with $(qubits) qubits:")
    foreach(state) do (key, value)
        basis = [Int(!iszero(key & mask)) for mask in state.masks]
        if isreal(value)
            print(io, " ", (real(value) > 0 ? "+" : ""), round(real(value); sigdigits=6), "|", join(basis), "⟩")
        else
            print(io, " +(", round(value; sigdigits=6), ")|", join(basis), "⟩")
        end
        if qubits > 4 || length(state) > 4
            print('\n')
        end
    end
    return nothing
end

default_droptol(::Type{T}) where {T<:Complex} = default_droptol(real(T))
default_droptol(::Type{T}) where {T<:Union{Integer,Rational}} = zero(T)
default_droptol(::Type{T}) where {T<:Number} = eps(T)^(2//3)

function sorted_merge!(t₁::AbstractVector{Pair{K,V₁}}, t₂::AbstractVector{Pair{K,V₂}}; droptol) where {K,V₁,V₂}
    # Merges two tables assuming they are sorted
    V = promote_type(V₁, V₂)
    i₁ = firstindex(t₁)
    @label beginning
    @inbounds if !isempty(t₂)
        i₂ = firstindex(t₂)
        k₂, v₂ = t₂[i₂]
        while i₁ ≤ lastindex(t₁)
            k₁, v₁ = t₁[i₁]
            if k₁ > k₂
                insert!(t₁, i₁, k₂ => v₂)
                popat!(t₂, i₂)
                @goto beginning
            elseif k₁ == k₂
                v = v₁ + v₂
                if isapprox(v, zero(V); atol=droptol)
                    # drop both entries
                    popat!(t₁, i₁)
                    popat!(t₂, i₂)
                else
                    t₁[i₁] = k₁ => v
                    popat!(t₂, i₂)
                    i₁ += 1
                end
                @goto beginning
            end
            i₁ += 1
        end
        # Traversed all of `t₁`, just append
        append!(t₁, t₂)
    end
    return t₁
end

function sorted_merge!(
    first_state::SparseState{K,V₁}, second_state::SparseState{K,V₂}; droptol=default_droptol(promote_type(V₁, V₂))
) where {K,V₁,V₂}
    @boundscheck num_qubits(first_state) == num_qubits(second_state) ||
        throw(ArgumentError("States do not have the same number of qubits"))
    sorted_merge!(table(first_state), table(second_state); droptol)
    return first_state
end

function Base.isapprox(first_state::SparseState{K,V₁}, second_state::SparseState{K,V₂}; kwargs...) where {K,V₁,V₂}
    @boundscheck num_qubits(first_state) == num_qubits(second_state) ||
        throw(ArgumentError("States do not have the same number of qubits"))
    # Compare element-wise
    t₁, t₂ = table(first_state), table(second_state)
    i₁, i₂ = firstindex(t₁), firstindex(t₂)
    @label beginning
    while i₁ ≤ lastindex(t₁)
        k₁, v₁ = t₁[i₁]
        while i₂ ≤ lastindex(t₂)
            k₂, v₂ = t₂[i₂]
            if k₂ < k₁
                isapprox(v₂, zero(V₂); kwargs...) || return false
                i₂ += 1
            elseif k₁ == k₂
                isapprox(v₁, v₂; kwargs...) || return false
                i₁ += 1
                i₂ += 1
                @goto beginning
            else
                break
            end
        end
        isapprox(v₁, zero(V₁); kwargs...) || return false
        i₁ += 1
    end
    return true
end

LinearAlgebra.lmul!(α::Number, state::SparseState) = rmul!(state, α)

function LinearAlgebra.rmul!(state::SparseState, α::Number)
    (; table) = state
    @inbounds for i in eachindex(table)
        s, v = table[i]
        table[i] = s => α * v
    end
    return state
end

Base.:+(first_state::SparseState, second_state::SparseState) = sorted_merge!(copy(first_state), copy(second_state))
Base.:-(state::SparseState) = -one(valtype(state)) * state
Base.:-(first_state::SparseState, second_state::SparseState) = first_state + (-second_state)
Base.:*(α::Number, state::SparseState) = state * α
Base.:*(state::SparseState, α::Number) = rmul!(copy(state), α)
Base.:/(state::SparseState, α::Number) = state * inv(α)

function LinearAlgebra.kron(first_state::SparseState{K,V₁}, second_state::SparseState{K,V₂}) where {K,V₁,V₂}
    new_table = [
        s₁ | (s₂ << num_qubits(first_state)) => v₁ * v₂ for (s₁, v₁) in first_state for (s₂, v₂) in second_state
    ]
    return SparseState(sort!(new_table; by=first), num_qubits(first_state) + num_qubits(second_state))
end

LinearAlgebra.kron(states::SparseState...) = foldl(kron, states)

LinearAlgebra.norm(state::SparseState, args...) = norm(values(state), args...)

LinearAlgebra.normalize!(state::SparseState, args...) = rmul!(state, 1 / norm(state, args...))

function LinearAlgebra.dot(first_state::SparseState{K}, second_state::SparseState{K}) where {K}
    @boundscheck num_qubits(first_state) == num_qubits(second_state) ||
        throw(ArgumentError("States do not have the same number of qubits"))
    return sum(first_state) do (s, v₁)
        v₂ = get(second_state, s, zero(valtype(second_state)))
        return conj(v₁) * v₂
    end
end

function expectation(state::SparseState, i::Int)
    m = state.masks[i]
    return sum(!iszero(s & m) * abs2(v) for (s, v) in state)
end

expectation(state::SparseState, indices) = map(i -> expectation(state, i), indices)
