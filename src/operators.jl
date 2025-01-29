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

parity(i::Integer) = isodd(count_ones(i))

conditional_minus(b::Bool) = (1 - 2 * b)
conditional_conj(x::Complex{T}, b::Bool) where {T} = Complex{T}(real(x), conditional_minus(b) * imag(x))
conditional_conj(x::Number, b::Bool) = x

function apply!(gate::I, state::SparseState; kwargs...)
    return state
end

function apply!(gate::X, state::SparseState; kwargs...)
    (; table, masks) = state
    m = mapreduce(inds -> masks[only(inds)], ⊻, support(gate))
    @inbounds for i in eachindex(table)
        s, v = table[i]
        table[i] = s ⊻ m => v
    end
    return state
end

function apply!(gate::Y, state::SparseState; kwargs...)
    (; table, masks) = state
    c = im^length(support(gate))
    m = mapreduce(inds -> masks[only(inds)], ⊻, support(gate))
    @inbounds for i in eachindex(table)
        s, v = table[i]
        table[i] = s ⊻ m => c * conditional_minus(parity(s & m)) * v
    end
    return state
end

function apply!(gate::Z, state::SparseState; kwargs...)
    (; table, masks) = state
    m = mapreduce(inds -> masks[only(inds)], ⊻, support(gate))
    @inbounds for i in eachindex(table)
        s, v = table[i]
        table[i] = s => conditional_minus(parity(s & m)) * v
    end
    return state
end

function apply!(gate::S, state::SparseState; kwargs...)
    (; table, masks) = state
    m = mapreduce(inds -> masks[only(inds)], ⊻, support(gate))
    @inbounds for i in eachindex(table)
        s, v = table[i]
        table[i] = s => im^count_ones(s & m) * v
    end
    return state
end

function apply!(gate::T, state::SparseState; kwargs...)
    (; table, masks) = state
    c = convert(valtype(state), (1 + im) / sqrt(2))
    m = mapreduce(inds -> masks[only(inds)], ⊻, support(gate))
    @inbounds for i in eachindex(table)
        s, v = table[i]
        table[i] = s => c^count_ones(s & m) * v
    end
    return state
end

function apply!(gate::H, state::SparseState; droptol=default_droptol(keytype(state)), kwargs...)
    (; table, masks) = state
    sort!(table; by=first)
    new_table = similar(table, 0)
    # Precompute the normalization factor
    c = convert(valtype(state), 1 / √2)
    for (i,) in support(gate)
        empty!(new_table)
        sizehint!(new_table, length(table))
        m = masks[i]
        @inbounds for i in eachindex(table)
            s, v = table[i]
            # One state just changes sign, it remains in the current table
            table[i] = s => c * conditional_minus(parity(s & m)) * v
            # Save flipped state to new table
            push!(new_table, s ⊻ m => c * v)
        end
        # Merge new table and combine it with old one
        sort!(new_table; by=first)
        sorted_merge!(table, new_table; droptol)
    end
    return state
end

function apply!(gate::U, state::SparseState; droptol=default_droptol(keytype(state)), kwargs...)
    (; table, masks) = state
    sort!(table; by=first)
    new_table = similar(table, 0)
    # Precompute the different factors for the gate to be in the form `[α -β'; β α']`
    (; θ, ϕ, λ) = gate
    a, b = exp(-(im / 2) * ϕ), exp(-(im / 2) * λ)
    s, c = sincos(θ / 2)
    α, β = convert(valtype(state), a * b * c), convert(valtype(state), a' * b * s)
    for (i,) in support(gate)
        empty!(new_table)
        sizehint!(new_table, length(table))
        m = masks[i]
        @inbounds for i in eachindex(table)
            s, v = table[i]
            # One state just changes sign, it remains in the current table
            table[i] = s => conditional_conj(α, !iszero(s & m)) * v
            # Save flipped state to new table
            push!(new_table, s ⊻ m => -im * conditional_conj(im * β, !iszero(s & m)) * v)
        end
        # Merge new table and combine it with old one
        sort!(new_table; by=first)
        sorted_merge!(table, new_table; droptol)
    end
    return state
end

function apply!(gate::RX, state::SparseState; kwargs...)
    (; θ) = gate
    return apply!(U(support(gate); θ=θ, ϕ=-(π / 2), λ=(+(π / 2))), state)
end

function apply!(gate::RY, state::SparseState; kwargs...)
    (; θ) = gate
    return apply!(U(support(gate); θ=θ, ϕ=zero(θ), λ=zero(θ)), state)
end

function apply!(gate::RZ, state::SparseState; kwargs...)
    (; table, masks) = state
    (; θ) = gate
    z = convert(valtype(state), exp(-(im / 2) * θ))
    m = mapreduce(inds -> masks[only(inds)], ⊻, support(gate))
    @inbounds for i in eachindex(table)
        s, v = table[i]
        table[i] = s => conditional_conj(z, parity(s & m)) * v
    end
    return state
end

function apply!(gate::CX, state::SparseState; kwargs...)
    (; table, masks) = state
    for (i, j) in support(gate)
        mᵢ, mⱼ = masks[i], masks[j]
        @inbounds for i in eachindex(table)
            s, v = table[i]
            s ⊻= !iszero(s & mᵢ) * mⱼ
            table[i] = s => v
        end
    end
    return state
end

function apply!(gate::CY, state::SparseState; kwargs...)
    (; table, masks) = state
    for (i, j) in support(gate)
        mᵢ, mⱼ = masks[i], masks[j]
        @inbounds for i in eachindex(table)
            s, v = table[i]
            c = !iszero(s & mᵢ)
            u = c * mⱼ
            v *= im^c * conditional_minus(!iszero(s & u))
            s ⊻= u
            table[i] = s => v
        end
    end
    return state
end

function apply!(gate::CZ, state::SparseState; kwargs...)
    (; table, masks) = state
    for (i, j) in support(gate)
        mᵢ, mⱼ = masks[i], masks[j]
        @inbounds for i in eachindex(table)
            s, v = table[i]
            v *= conditional_minus(!iszero(s & mᵢ) & !iszero(s & mⱼ))
            table[i] = s => v
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
        @inbounds for i in eachindex(table)
            s, v = table[i]
            u = (s & mᵢ) ⊻ ((s & mⱼ) >> shift)
            s ⊻= u | (u << shift)
            table[i] = s => v
        end
    end
    return state
end

function apply!(gate::CCX, state::SparseState; kwargs...)
    (; table, masks) = state
    for (i, j, k) in support(gate)
        mᵢ, mⱼ, mₖ = masks[i], masks[j], masks[k]
        @inbounds for i in eachindex(table)
            s, v = table[i]
            s ⊻= (!iszero(s & mᵢ) & !iszero(s & mⱼ)) * mₖ
            table[i] = s => v
        end
    end
    return state
end

function apply!(gate::CCY, state::SparseState; kwargs...)
    (; table, masks) = state
    for (i, j, k) in support(gate)
        mᵢ, mⱼ, mₖ = masks[i], masks[j], masks[k]
        @inbounds for i in eachindex(table)
            s, v = table[i]
            c = !iszero(s & mᵢ) & !iszero(s & mⱼ)
            u = c * mₖ
            v *= im^c * conditional_minus(!iszero(s & u))
            table[i] = s => v
        end
    end
    return state
end

function apply!(gate::CCZ, state::SparseState; kwargs...)
    (; table, masks) = state
    for (i, j, k) in support(gate)
        mᵢ, mⱼ, mₖ = masks[i], masks[j], masks[k]
        @inbounds for i in eachindex(table)
            s, v = table[i]
            v *= conditional_minus(!iszero(s & mᵢ) & !iszero(s & mⱼ) & !iszero(s & mₖ))
            table[i] = s => v
        end
    end
    return state
end
