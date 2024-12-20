module SparseStates

using LinearAlgebra

export SparseState
export num_qubits, expectation
export AbstractOperator, Operator, SuperOperator
export support, apply, apply!
export X, Y, Z, H, S, T, CX, CNOT, CY, CZ, SWAP, CCX, CCNOT, CCY, CCZ
export DepolarizingChannel, Reset, Measure
export Circuit
export pauli_combinations, pauli_strings
export pauli_decomposition, append_operators, drop_error_operators, error_locations, insert_error_operators


include("states.jl")
include("abstractoperators.jl")
include("operators.jl")
include("circuits.jl")
include("superoperators.jl")
include("utils.jl")

end
