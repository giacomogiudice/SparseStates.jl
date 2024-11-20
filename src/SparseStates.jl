module SparseStates

using LinearAlgebra

export SparseState
export num_qubits, expectation
export AbstractOperator, Operator, SuperOperator
export support, apply
export X, Y, Z, H, S, CX, CNOT, CY, CZ, CCX, CCNOT, CCY, CCZ
export DepolarizingChannel, Reset, Measure
export Circuit
export pauli_combinations, pauli_strings, append_operators
export pauli_decomposition

include("states.jl")
include("abstractoperators.jl")
include("operators.jl")
include("circuits.jl")
include("superoperators.jl")
include("utils.jl")

end
