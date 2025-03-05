# Use the `@operator` macro to define common constructor patterns
@operator I 1
@operator X 1
@operator Y 1
@operator Z 1

@operator S 1
@operator H 1
@operator T 1

@operator U{T<:Number} 1 θ::T = 0.0 ϕ::T = 0.0 λ::T = 0.0
@operator RX{T<:Number} 1 θ::T = 0.0
@operator RY{T<:Number} 1 θ::T = 0.0
@operator RZ{T<:Number} 1 θ::T = 0.0

@operator CX 2
@operator CY 2
@operator CZ 2

@operator SWAP 2
@operator CCX 3
@operator CCY 3
@operator CCZ 3

# Aliases
CNOT = CX
CCNOT = CCX

for G in (I, X, Y, Z, H, CX, CY, CZ, SWAP, CCX, CCY, CCZ)
    @eval apply!(gate::AdjointOperator{<:$G}, state::SparseState; kwargs...) = apply!(parent(gate), state; kwargs...)
end

parity(i::Integer) = isodd(count_ones(i))

conditional_minus(b::Bool) = (1 - 2 * b)

conditional_conj(x::Complex, b::Bool) = complex(real(x), conditional_minus(b) * imag(x))
conditional_conj(x::Number, b::Bool) = x

function apply!(gate::I, state::SparseState; kwargs...)
    return state
end

function apply!(gate::X, state::SparseState; kwargs...)
    (; table, masks) = state
    m = mapreduce(inds -> masks[only(inds)], ⊻, support(gate); init=zero(keytype(state)))
    @inbounds for n in eachindex(table)
        s, v = table[n]
        table[n] = s ⊻ m => v
    end
    return state
end

function apply!(gate::Y, state::SparseState; kwargs...)
    (; table, masks) = state
    z = im^length(support(gate))
    m = mapreduce(inds -> masks[only(inds)], ⊻, support(gate); init=zero(keytype(state)))
    @inbounds for n in eachindex(table)
        s, v = table[n]
        table[n] = s ⊻ m => z * conditional_minus(parity(s & m)) * v
    end
    return state
end

function apply!(gate::Z, state::SparseState; kwargs...)
    (; table, masks) = state
    m = mapreduce(inds -> masks[only(inds)], ⊻, support(gate); init=zero(keytype(state)))
    @inbounds for n in eachindex(table)
        s, v = table[n]
        table[n] = s => conditional_minus(parity(s & m)) * v
    end
    return state
end

function apply!(gate::Union{S,AdjointOperator{S}}, state::SparseState; kwargs...)
    (; table, masks) = state
    m = mapreduce(inds -> masks[only(inds)], ⊻, support(gate); init=zero(keytype(state)))
    z = im
    (gate isa AdjointOperator) && (z = conj(z))
    @inbounds for n in eachindex(table)
        s, v = table[n]
        table[n] = s => z^count_ones(s & m) * v
    end
    return state
end

function apply!(gate::Union{T,AdjointOperator{T}}, state::SparseState; kwargs...)
    (; table, masks) = state
    z = convert(valtype(state), (1 + im) / sqrt(2))
    (gate isa AdjointOperator) && (z = conj(z))
    m = mapreduce(inds -> masks[only(inds)], ⊻, support(gate); init=zero(keytype(state)))
    @inbounds for n in eachindex(table)
        s, v = table[n]
        table[n] = s => z^count_ones(s & m) * v
    end
    return state
end

function apply!(gate::H, state::SparseState; droptol=default_droptol(keytype(state)), kwargs...)
    (; table, masks) = state
    sort!(table; by=first)
    # Precompute the normalization factor
    z = convert(valtype(state), 1 / √2)
    for (i,) in support(gate)
        new_table = similar(table)
        m = masks[i]
        @inbounds for n in eachindex(table)
            s, v = table[n]
            # One state just changes sign, it remains in the current table
            table[n] = s => z * conditional_minus(parity(s & m)) * v
            # Save flipped state to new table
            new_table[n] = s ⊻ m => z * v
        end
        # Merge new table and combine it with old one
        sort!(new_table; by=first)
        sorted_merge!(table, new_table; droptol)
    end
    return state
end

function apply!(gate::U, state::SparseState; droptol=default_droptol(valtype(state)), kwargs...)
    (; table, masks) = state
    sort!(table; by=first)
    # Precompute the different factors for the gate to be in the form `[α -β'; β α']`
    (; θ, ϕ, λ) = gate
    a, b = cis(-ϕ / 2), cis(-λ / 2)
    s, c = sincos(θ / 2)
    α, β = convert(valtype(state), a * b * c), convert(valtype(state), a' * b * s)
    for (i,) in support(gate)
        new_table = similar(table)
        m = masks[i]
        @inbounds for n in eachindex(table)
            s, v = table[n]
            # One state just changes sign, it remains in the current table
            table[n] = s => conditional_conj(α, !iszero(s & m)) * v
            # Save flipped state to new table
            new_table[n] = s ⊻ m => -im * conditional_conj(im * β, !iszero(s & m)) * v
        end
        # Merge new table and combine it with old one
        sort!(new_table; by=first)
        sorted_merge!(table, new_table; droptol)
    end
    return state
end

function apply!(gate::AdjointOperator{<:U}, state::SparseState; kwargs...)
    (; θ, ϕ, λ) = parent(gate)
    return apply!(U(support(gate); θ=-θ, ϕ=-λ, λ=-ϕ), state; kwargs...)
end

function apply!(gate::RX, state::SparseState; kwargs...)
    (; θ) = gate
    return apply!(U(support(gate); θ=θ, ϕ=(-π / 2), λ=(+π / 2)), state; kwargs...)
end

function apply!(gate::AdjointOperator{<:RX}, state::SparseState; kwargs...)
    (; θ) = parent(gate)
    return apply!(RX(support(gate); θ=-θ), state; kwargs...)
end

function apply!(gate::RY, state::SparseState; kwargs...)
    (; θ) = gate
    return apply!(U(support(gate); θ=θ, ϕ=zero(θ), λ=zero(θ)), state; kwargs...)
end

function apply!(gate::AdjointOperator{<:RY}, state::SparseState; kwargs...)
    (; θ) = parent(gate)
    return apply!(RY(support(gate); θ=-θ), state; kwargs...)
end

function apply!(gate::RZ, state::SparseState; kwargs...)
    (; table, masks) = state
    (; θ) = gate
    z = convert(valtype(state), cis(-θ / 2))
    m = mapreduce(inds -> masks[only(inds)], ⊻, support(gate))
    @inbounds for n in eachindex(table)
        s, v = table[n]
        table[n] = s => conditional_conj(z, parity(s & m)) * v
    end
    return state
end

function apply!(gate::AdjointOperator{<:RZ}, state::SparseState; kwargs...)
    (; θ) = parent(gate)
    return apply!(RZ(support(gate); θ=-θ), state; kwargs...)
end

function apply!(gate::CX, state::SparseState; kwargs...)
    (; table, masks) = state
    for (i, j) in support(gate)
        mᵢ, mⱼ = masks[i], masks[j]
        @inbounds for n in eachindex(table)
            s, v = table[n]
            s ⊻= !iszero(s & mᵢ) * mⱼ
            table[n] = s => v
        end
    end
    return state
end

function apply!(gate::CY, state::SparseState; kwargs...)
    (; table, masks) = state
    for (i, j) in support(gate)
        mᵢ, mⱼ = masks[i], masks[j]
        @inbounds for n in eachindex(table)
            s, v = table[n]
            c = !iszero(s & mᵢ)
            u = c * mⱼ
            v *= im^c * conditional_minus(!iszero(s & u))
            s ⊻= u
            table[n] = s => v
        end
    end
    return state
end

function apply!(gate::CZ, state::SparseState; kwargs...)
    (; table, masks) = state
    for (i, j) in support(gate)
        mᵢ, mⱼ = masks[i], masks[j]
        @inbounds for n in eachindex(table)
            s, v = table[n]
            v *= conditional_minus(!iszero(s & mᵢ) & !iszero(s & mⱼ))
            table[n] = s => v
        end
    end
    return state
end

function apply!(gate::SWAP, state::SparseState; kwargs...)
    (; table, masks) = state
    for (i, j) in support(gate)
        if j < i
            i, j = j, i
        end
        shift = convert(keytype(state), j - i)
        mᵢ, mⱼ = masks[i], masks[j]
        @inbounds for n in eachindex(table)
            s, v = table[n]
            u = (s & mᵢ) ⊻ ((s & mⱼ) >> shift)
            s ⊻= u | (u << shift)
            table[n] = s => v
        end
    end
    return state
end

function apply!(gate::CCX, state::SparseState; kwargs...)
    (; table, masks) = state
    for (i, j, k) in support(gate)
        mᵢ, mⱼ, mₖ = masks[i], masks[j], masks[k]
        @inbounds for n in eachindex(table)
            s, v = table[n]
            s ⊻= (!iszero(s & mᵢ) & !iszero(s & mⱼ)) * mₖ
            table[n] = s => v
        end
    end
    return state
end

function apply!(gate::CCY, state::SparseState; kwargs...)
    (; table, masks) = state
    for (i, j, k) in support(gate)
        mᵢ, mⱼ, mₖ = masks[i], masks[j], masks[k]
        @inbounds for n in eachindex(table)
            s, v = table[n]
            c = !iszero(s & mᵢ) & !iszero(s & mⱼ)
            u = c * mₖ
            v *= im^c * conditional_minus(!iszero(s & u))
            s ⊻= u
            table[n] = s => v
        end
    end
    return state
end

function apply!(gate::CCZ, state::SparseState; kwargs...)
    (; table, masks) = state
    for (i, j, k) in support(gate)
        mᵢ, mⱼ, mₖ = masks[i], masks[j], masks[k]
        @inbounds for n in eachindex(table)
            s, v = table[n]
            v *= conditional_minus(!iszero(s & mᵢ) & !iszero(s & mⱼ) & !iszero(s & mₖ))
            table[n] = s => v
        end
    end
    return state
end
