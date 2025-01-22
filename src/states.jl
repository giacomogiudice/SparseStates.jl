using Base: Callable

const DEFAULT_KEYTYPE = UInt64
const DEFAULT_ELTYPE = Complex{Float64}

struct SparseState{K,V} <: AbstractDict{K,V}
    table::Vector{Pair{K,V}}
    masks::Vector{K}
end

function SparseState(table::AbstractVector{Pair{K,V}}, qubits::Int) where {K,V}
    masks = map(n -> one(K) << n, 0:(qubits - 1))
    return SparseState{K,V}(sort(table; by=first), masks)
end

SparseState(generator, qubits::Int) = SparseState(sort!(vec(collect(generator)); by=first), qubits)

SparseState{K,V}(qubits::Int) where {K,V} = SparseState([zero(K) => one(V)], qubits)
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

convert_to_keytype(state::SparseState, key) = convert(keytype(state), key)

function convert_to_keytype(state::SparseState, key::AbstractString)
    length(key) == num_qubits(state) ||
        throw(ArgumentError("Key string length does not correspond to the number of qubits"))
    (; masks) = state
    parsed_key = mapreduce(|, key, masks) do c, m
        if c == '0'
            return zero(m)
        elseif c == '1'
            return m
        else
            throw(ArgumentError("key $(key) is expected to only have 0s and 1s"))
        end
    end
    return parsed_key
end

function Base.haskey(state::SparseState, key)
    key = convert_to_keytype(state, key)
    (; table) = state
    i = searchsortedfirst(table, key; by=first)
    return i ≤ length(table) && first(table[i]) == key
end

function Base.get(state::SparseState, key, default)
    key = convert_to_keytype(state, key)
    (; table) = state
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
    key = convert_to_keytype(state, key)
    (; table) = state
    i = searchsortedfirst(table, key; by=first)
    if i ≤ length(table)
        pair = table[i]
        if first(pair) == key
            return last(pair)
        end
    end
    insert!(table, i, default())
    return default
end

Base.getindex(state::SparseState, key::AbstractString) = getindex(state, convert_to_keytype(state, key))

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
default_droptol(::Type{T}) where {T<:Number} = sqrt(eps(T))

function sorted_merge!(t₁::AbstractVector{Pair{K,V₁}}, t₂::AbstractVector{Pair{K,V₂}}; droptol) where {K,V₁,V₂}
    # Merges two tables assuming they are sorted
    V = promote_type(V₁, V₂)
    i₁ = firstindex(t₁)
    @label beginning
    @inbounds if !isempty(t₂)
        i₂ = firstindex(t₂)
        k₂, v₂ = t₂[i₂]
        while i₁ <= lastindex(t₁)
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

function Base.:+(first_state::SparseState{K}, second_state::SparseState{K}) where {K}
    @boundscheck num_qubits(first_state) == num_qubits(second_state) ||
        throw(ArgumentError("States do not have the same number of qubits"))
    droptol = default_droptol(promote_type(valtype(first_state), valtype(second_state)))
    new_table = sorted_merge!(copy(table(first_state)), copy(table(second_state)); droptol)
    return SparseState(new_table, num_qubits(first_state))
end

function Base.:*(α::Number, state::SparseState)
    return SparseState(Dict(s => α * v for (s, v) in state), num_qubits(state))
end

Base.:*(state::SparseState, α::Number) = α * state
Base.:-(state::SparseState) = -one(valtype(state)) * state
Base.:-(first_state::SparseState, second_state::SparseState) = first_state + (-second_state)
Base.:/(state::SparseState, α::Number) = inv(α) * state

function LinearAlgebra.kron(first_state::SparseState{K,V₁}, second_state::SparseState{K,V₂}) where {K,V₁,V₂}
    new_table = [
        s₁ | (s₂ << num_qubits(first_state)) => v₁ * v₂ for (s₁, v₁) in first_state for (s₂, v₂) in second_state
    ]
    return SparseState(sort!(new_table; by=first), num_qubits(first_state) + num_qubits(second_state))
end

LinearAlgebra.kron(states::SparseState...) = foldl(kron, states)

LinearAlgebra.norm(state::SparseState, args...) = norm(values(state), args...)

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
