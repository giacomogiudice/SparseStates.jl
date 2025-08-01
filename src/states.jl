using Base: RefValue

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
            throw(ArgumentError("key $(key) is expected to only have 0s or 1s"))
        end
    end
end

struct SparseState{K,V} <: AbstractDict{K,V}
    table::Vector{Pair{K,V}}
    masks::Vector{K}
    refsorted::RefValue{Bool}
end

function SparseState{K,V}(pairs, qubits::Int) where {K,V}
    qubits > 8 * sizeof(K) &&
        throw(ArgumentError("The number of qubits $qubits exceeds the number of bits in the keytype $K"))
    masks = map(n -> one(K) << n, 0:(qubits - 1))
    iter = Iterators.map(pairs) do (key, val)
        return parse_key(key, masks) => val
    end
    return SparseState{K,V}(sort!(vec(collect(iter)); by=first), masks, Ref(true))
end

SparseState{K}(pairs, qubits::Int) where {K} = SparseState{K,DEFAULT_ELTYPE}(pairs, qubits)
SparseState(pairs, qubits::Int) = SparseState{DEFAULT_KEYTYPE}(pairs, qubits)
SparseState(pairs::AbstractVector{Pair{K,V}}, qubits::Int) where {K<:Integer,V} = SparseState{K,V}(pairs, qubits)

SparseState{K,V}(pair::Pair, qubits::Int) where {K,V} = SparseState{K,V}([pair], qubits)
SparseState(pair::Pair, qubits::Int) = SparseState([pair], qubits)

SparseState{K,V}(qubits::Int) where {K,V} = SparseState{K,V}([zero(K) => one(V)], qubits)
SparseState{K}(qubits::Int) where {K} = SparseState{K,DEFAULT_ELTYPE}(qubits)
SparseState(qubits::Int) = SparseState{DEFAULT_KEYTYPE}(qubits)

num_qubits(state::SparseState) = length(state.masks)

Base.length(state::SparseState) = length(state.table)
Base.empty(state::SparseState, K, V) = SparseState{K,V}([], state.masks, true)
Base.copy(state::SparseState) = SparseState(copy(state.table), copy(state.masks), Ref(state.refsorted[]))
Base.sizehint!(state::SparseState, n::Integer) = sizehint!(state.table, n)
Base.iterate(state::SparseState, args...) = iterate(state.table, args...)
Base.pairs(state::SparseState) = state.table

Base.issorted(state::SparseState) = state.refsorted[]
setsorted!(state::SparseState, val) = setindex!(state.refsorted, Bool(val))

function Base.sort!(state::SparseState; force::Bool=true)
    !force && issorted(state) && return state
    sort!(state.table; by=first)
    setsorted!(state, true)
    return state
end

function Base.haskey(state::SparseState, key)
    sort!(state)
    (; table, masks) = state
    key = parse_key(key, masks)
    n = searchsortedfirst(table, key; by=first)
    return n ≤ length(table) && first(table[n]) == key
end

function Base.get(state::SparseState, key, default)
    sort!(state)
    (; table, masks) = state
    key = parse_key(key, masks)
    n = searchsortedfirst(table, key; by=first)
    if n ≤ length(table)
        pair = table[n]
        if first(pair) == key
            return last(pair)
        end
    end
    return default
end

function Base.get!(default::Function, state::SparseState, key)
    sort!(state)
    (; table, masks) = state
    key = parse_key(key, masks)
    n = searchsortedfirst(table, key; by=first)
    if n ≤ length(table)
        pair = table[n]
        if first(pair) == key
            return last(pair)
        end
    end
    val = convert!(valtype(state), default())
    insert!(table, n, val)
    return val
end

function Base.getindex(state::SparseState, key::AbstractString)
    (; masks) = state
    return getindex(state, parse_key(key, masks))
end

function Base.setindex!(state::SparseState, key, value)
    sort!(state)
    key = convert_to_keytype(state, key)
    value = convert(valtype(state), value)
    table = table(state)
    n = searchsortedfirst(table, key; by=first)
    if n ≤ length(table)
        pair = table[n]
        if first(pair) == key
            table[n] = key => value
        end
    end
    insert!(table, n, key => value)
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
        (qubits > 4 || length(state) > 4) && println(io)
    end
    return nothing
end

default_droptol(::Type{T}) where {T<:Complex} = default_droptol(real(T))
default_droptol(::Type{T}) where {T<:Union{Integer,Rational}} = zero(T)
default_droptol(::Type{T}) where {T<:Number} = eps(T)^(2//3)

function sorted_merge!(t₁::AbstractVector{Pair{K,V₁}}, t₂::AbstractVector{Pair{K,V₂}}; droptol) where {K,V₁,V₂}
    # Merges two tables in linear time, assuming they are sorted
    V = promote_type(V₁, V₂)
    n₁ = firstindex(t₁)
    @label beginning
    @inbounds if !isempty(t₂)
        n₂ = firstindex(t₂)
        k₂, v₂ = t₂[n₂]
        while n₁ ≤ lastindex(t₁)
            k₁, v₁ = t₁[n₁]
            if k₁ > k₂
                insert!(t₁, n₁, popat!(t₂, n₂))
                @goto beginning
            elseif k₁ == k₂
                v = v₁ + v₂
                if isapprox(v, zero(V); atol=droptol)
                    # drop both entries
                    popat!(t₁, n₁)
                    popat!(t₂, n₂)
                else
                    t₁[n₁] = k₁ => v
                    popat!(t₂, n₂)
                    n₁ += 1
                end
                @goto beginning
            end
            n₁ += 1
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
    sort!(first_state)
    sort!(second_state)
    sorted_merge!(first_state.table, second_state.table; droptol)
    return first_state
end

function Base.isapprox(first_state::SparseState{K,V₁}, second_state::SparseState{K,V₂}; kwargs...) where {K,V₁,V₂}
    @boundscheck num_qubits(first_state) == num_qubits(second_state) ||
        throw(ArgumentError("States do not have the same number of qubits"))
    sort!(first_state)
    sort!(second_state)
    # Compare element-wise, assuming sorted tables
    t₁, t₂ = first_state.table, second_state.table
    n₁, n₂ = firstindex(t₁), firstindex(t₂)
    @label beginning
    @inbounds while n₁ ≤ lastindex(t₁)
        k₁, v₁ = t₁[n₁]
        while n₂ ≤ lastindex(t₂)
            k₂, v₂ = t₂[n₂]
            if k₂ < k₁
                isapprox(v₂, zero(V₂); kwargs...) || return false
                n₂ += 1
            elseif k₁ == k₂
                isapprox(v₁, v₂; kwargs...) || return false
                n₁ += 1
                n₂ += 1
                @goto beginning
            else
                break
            end
        end
        isapprox(v₁, zero(V₁); kwargs...) || return false
        n₁ += 1
    end
    return true
end

LinearAlgebra.lmul!(α::Number, state::SparseState) = rmul!(state, α)

function LinearAlgebra.rmul!(state::SparseState, α::Number)
    (; table) = state
    @inbounds for n in eachindex(table)
        s, v = table[n]
        table[n] = s => α * v
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
    n₁, n₂ = num_qubits(first_state), num_qubits(second_state)
    new_table = [s₁ | (s₂ << n₁) => v₁ * v₂ for (s₁, v₁) in first_state for (s₂, v₂) in second_state]
    return SparseState(new_table, n₁ + n₂)
end

LinearAlgebra.kron(states::SparseState...) = foldl(kron, states)

LinearAlgebra.norm(state::SparseState) = √sum(abs2, values(state))

LinearAlgebra.normalize!(state::SparseState) = rmul!(state, 1 / norm(state))

function LinearAlgebra.dot(first_state::SparseState, second_state::SparseState)
    @boundscheck num_qubits(first_state) == num_qubits(second_state) ||
        throw(ArgumentError("States do not have the same number of qubits"))
    sort!(first_state)
    sort!(second_state)
    # Add common elements, using the fact that the tables are sorted
    ret = zero(promote_type(valtype(first_state), valtype(second_state)))
    t₁, t₂ = first_state.table, second_state.table
    n₂ = firstindex(t₂)
    @inbounds for n₁ in eachindex(t₁)
        k₁, v₁ = t₁[n₁]
        k₂, v₂ = t₂[n₂]
        while n₂ ≤ lastindex(t₂)
            k₂, v₂ = t₂[n₂]
            if k₂ < k₁
                n₂ += 1
            else
                break
            end
        end
        if k₁ == k₂
            ret += conj(v₁) * v₂
        end
    end
    return ret
end

function expectation(state::SparseState{K,V}, i::Int) where {K,V}
    m = state.masks[i]
    upper, lower = zero(real(V)), zero(real(V))
    @inbounds for (s, v) in pairs(state)
        a = abs2(v)
        upper += !iszero(s & m) * a
        lower += a
    end
    return upper / √lower
end

expectation(state::SparseState, indices) = map(i -> expectation(state, i), indices)
