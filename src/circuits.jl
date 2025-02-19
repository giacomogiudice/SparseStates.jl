struct Circuit{T<:Tuple} <: AbstractOperator
    ops::T

    Circuit(ops::AbstractOperator...) = new{typeof(ops)}(ops)
end

Circuit(circuit::Circuit) = Circuit(parent(circuit)...)
Circuit(generator) = Circuit(reduce(*, generator))

Base.:*(x::AbstractOperator, y::AbstractOperator) = Circuit(x, y)
Base.:*(x::Circuit, y::AbstractOperator) = Circuit(parent(x)..., y)
Base.:*(x::AbstractOperator, y::Circuit) = Circuit(x, parent(y)...)
Base.:*(x::Circuit, y::Circuit) = Circuit(parent(x)..., parent(y)...)

Base.parent(circuit::Circuit) = circuit.ops
Base.length(circuit::Circuit) = length(parent(circuit))
Base.iterate(circuit::Circuit, args...) = iterate(parent(circuit), args...)
Base.firstindex(circuit::Circuit) = firstindex(parent(circuit))
Base.lastindex(circuit::Circuit) = lastindex(parent(circuit))
Base.getindex(circuit::Circuit, i) = getindex(parent(circuit), i)

support(circuit::Circuit) = mapreduce(support, vcat, parent(circuit); init=Tuple{Vararg{Int}}[])

function Base.show(io::IO, circuit::Circuit)
    if length(circuit) > 8
        print(io, circuit[begin], " * … * ", circuit[end])
    else
        print(io, join(parent(circuit), " * "))
    end
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", circuit::Circuit)
    print(io, "Circuit(")
    !isempty(circuit) && println(io)
    foreach(op -> println(io, "  ", op, ","), circuit)
    print(io, ")")
    return nothing
end

function apply!(circuit::Circuit, state::SparseState; kwargs...)
    foreach(parent(circuit)) do op
        apply!(op, state; kwargs...)
    end
    return state
end

pauli_combinations(qubits::Int=1) = Iterators.product(ntuple(_ -> (I, X, Y, Z), qubits)...)

function pauli_strings(qubits::Int=1)
    return Iterators.map(pauli_combinations(qubits)) do comb
        return Circuit(op(i) for (i, op) in enumerate(comb))
    end
end
