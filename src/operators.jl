# Use the `@operator` macro to define common constructor patterns
@operator(I, 1)
@operator(X, 1)
@operator(Y, 1)
@operator(Z, 1)

@operator(S, 1)
@operator(H, 1)

@operator(CX, 2)
@operator(CY, 2)
@operator(CZ, 2)
@operator(SWAP, 2)

@operator(CCX, 3)
@operator(CCY, 3)
@operator(CCZ, 3)

# Aliases
CNOT = CX
CCNOT = CCX

parity(i::Integer) = isodd(count_ones(i))

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
    m = mapreduce(inds -> masks[only(inds)], ⊻, support(gate))
    @inbounds for i in eachindex(table)
        s, v = table[i]
        table[i] = s ⊻ m => im * (1 - 2 * parity(s & m)) * v
    end
    return state
end

function apply!(gate::Z, state::SparseState; kwargs...)
    (; table, masks) = state
    m = mapreduce(inds -> masks[only(inds)], ⊻, support(gate))
    @inbounds for i in eachindex(table)
        s, v = table[i]
        table[i] = s => (1 - 2 * parity(s & m)) * v
    end
    return state
end

function apply!(gate::S, state::SparseState; kwargs...)
    (; table, masks) = state
    m = mapreduce(inds -> masks[only(inds)], ⊻, support(gate))
    @inbounds for i in eachindex(table)
        s, v = table[i]
        table[i] = s => im^parity(s & m) * v
    end
    return state
end

# function apply!(gate::H, state::SparseState; droptol=default_droptol(keytype(state)), kwargs...)
#     (; table, masks) = state
#     sort!(table; by=first)
#     # Precompute the normalization factor
#     n = convert(valtype(state), √2)
#     for (i,) in support(gate)
#         new_table = similar(table)
#         m = masks[i]
#         @inbounds for i in eachindex(table)
#             s, v = table[i]
#             # One state just changes sign, it remains in the current table
#             table[i] = s => (1 - 2 * parity(s & m)) * v / n
#             # Save flipped state to new table
#             new_table[i] = s ⊻ m => v / n
#         end
#         # Merge new table and combine it with old one
#         sort!(new_table; by=first)
#         sorted_merge!(table, new_table; droptol)
#     end
#     return state
# end

function apply!(gate::H, state::SparseState; droptol=default_droptol(keytype(state)), kwargs...)
    (; table, masks) = state
    sort!(table; by=first)
    new_table = similar(table, 0)
    # Precompute the normalization factor
    n = convert(valtype(state), √2)
    for (i,) in support(gate)
        empty!(new_table)
        sizehint!(new_table, length(table))
        m = masks[i]
        @inbounds for i in eachindex(table)
            s, v = table[i]
            # One state just changes sign, it remains in the current table
            table[i] = s => (1 - 2 * parity(s & m)) * v / n
            # Save flipped state to new table
            push!(new_table, s ⊻ m => v / n)
        end
        # Merge new table and combine it with old one
        sort!(new_table; by=first)
        sorted_merge!(table, new_table; droptol)
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
            v *= im^c * (1 - 2 * !iszero(s & u))
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
            v *= 1 - 2 * (!iszero(s & mᵢ) & !iszero(s & mⱼ))
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
            v *= im^c * (1 - 2 * !iszero(s & u))
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
            v *= 1 - 2 * (!iszero(s & mᵢ) & !iszero(s & mⱼ) & !iszero(s & mₖ))
            table[i] = s => v
        end
    end
    return state
end
