append_operators(op::AbstractOperator, mapping::Pair...) = append_operators(Circuit(op), mapping...)

function append_operators(circuit::Circuit, mapping::Pair...)
    result = Circuit()
    for op in circuit
        result *= op
        for (key, func) in mapping
            if typeof(op) == key || op == key
                result *= func(support(op))
            end
        end
    end
    return result
end

function pauli_decomposition(state::SparseState{K,V}; droptol=default_droptol(V)) where {K,V}
    n = num_qubits(state)
    all_coefficients = (op => real(dot(state, op, state)) / 2^n for op in pauli_strings(n))
    return Iterators.filter(((op, coeff),) -> abs(coeff) > droptol, all_coefficients)
end

function drop_error_operators(circuit::Circuit)
    return Circuit(Iterators.filter(op -> !(op isa DepolarizingChannel)), circuit)
end

function error_locations(circuit::Circuit)
    # Get location and support of all error channels
    locations = Pair{Int,Vector{Circuit}}[]
    for (loc, op) in enumerate(circuit)
        if op isa DepolarizingChannel
            for inds in support(op)
                combs = @view PAULI_COMBINATIONS[length(inds)][(begin + 1):end]
                push!(locations, loc => [mapreduce((O, i) -> O(i), *, ops, inds; init=Circuit()) for ops in combs])
            end
        end
    end
    return locations
end

function insert_error_operators(circuit::Circuit, pairs::Pair{Int,Circuit}...)
    new_circuit = Circuit()
    for (loc, op) in enumerate(circuit)
        if op isa DepolarizingChannel
            ind = findfirst(pair -> first(pair) == loc, pairs)
            if !isnothing(ind)
                new_circuit *= last(pairs[ind])
            end
        else
            new_circuit *= op
        end
    end
    return new_circuit
end
