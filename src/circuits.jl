struct Circuit <: AbstractOperator
    ops::Vector{<:AbstractOperator}

    Circuit(ops::AbstractVector) = new(Vector(ops))
    Circuit() = new(AbstractOperator[])
end

Circuit(circuit::Circuit) = Circuit(parent(circuit))
Circuit(ops::AbstractOperator...) = Circuit(reduce(*, ops))
Circuit(generator) = Circuit(reduce(*, generator))

Base.parent(circuit::Circuit) = circuit.ops
Base.length(circuit) = length(parent(circuit))
Base.iterate(circuit::Circuit, args...) = iterate(parent(circuit), args...)
Base.firstindex(circuit::Circuit) = firstindex(parent(circuit))
Base.lastindex(circuit::Circuit) = lastindex(parent(circuit))
Base.getindex(circuit::Circuit, i) = getindex(parent(circuit), i)
Base.setindex!(circuit::Circuit, i, val) = setindex!(parent(circuit), i, val)

support(circuit::Circuit) = union(map(op -> union(support(op)...), parent(circuit)))

function Base.show(io::IO, circuit::Circuit)
    if length(circuit) > 8
        print(io, circuit[begin], " * … * ", circuit[end])
    else
        print(io, join(parent(circuit), " * "))
    end
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", circuit::Circuit)
    println(io, "Circuit([")
    foreach(circuit) do op
        println(io, "  ", op, ",")
    end
    print(io, "])")
    return nothing
end

Base.:*(x::AbstractOperator, y::AbstractOperator) = Circuit([x, y])
Base.:*(x::Circuit, y::AbstractOperator) = Circuit(vcat(parent(x), y))
Base.:*(x::AbstractOperator, y::Circuit) = Circuit(vcat(x, parent(y)))
Base.:*(x::Circuit, y::Circuit) = Circuit(vcat(parent(x), parent(y)))

function apply!(circuit::Circuit, state::SparseState)
    foreach(parent(circuit)) do op
        apply!(op, state)
    end
    return state
end

pauli_combinations(qubits::Int=1) = Iterators.product(ntuple(_ -> (I, X, Y, Z), qubits)...)

function pauli_strings(qubits::Int=1)
    return Iterators.map(pauli_combinations(qubits)) do comb
        return Circuit(op(i) for (i, op) in enumerate(comb))
    end
end
