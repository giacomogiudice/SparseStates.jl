append_operators(op::AbstractOperator, mapping::Pair...) = append_operators(Circuit([op]), mapping...)

function append_operators(circuit::Circuit, mapping::Pair...)
    result = Circuit()
    for op in circuit
        result *= op
        for (key, func) in mapping
            if typeof(op) == key || op == key
                result *= func(op.support)
            end
        end
    end
    return result
end

function pauli_decomposition(state::SparseState{K,V}; tol=(√(eps(real(V))))) where {K,V}
    n = num_qubits(state)
    all_coefficients = (op => real(dot(state, op, state)) / 2^n for op in pauli_strings(n))
    return Iterators.filter(((op, coeff),) -> abs(coeff) > tol, all_coefficients)
end
