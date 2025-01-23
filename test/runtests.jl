using Test
using LinearAlgebra
using SparseStates

const KEY_TYPES = (UInt8, UInt16, UInt32, UInt64, UInt128)
const VAL_TYPES = (Float32, Float64, ComplexF32, ComplexF64)

@testset "SparseState initialization and operations" begin
    @testset "SparseState{$K,$V}" for (K, V) in Iterators.product(KEY_TYPES, VAL_TYPES)
        state = SparseState{K,V}(2)
        @test state isa AbstractDict
        @test (@inferred keytype(state)) == K && (@inferred valtype(state)) == V
        @test (@inferred num_qubits(state)) == 2
        @test (@inferred length(state)) == 1
        @test (@inferred haskey(state, 0))
        @test (@inferred haskey(state, "00"))
        @test (@inferred state[0]) ≈ 1
        @test (@inferred state["00"]) ≈ 1
        @test_throws ArgumentError state["000"]
        @test_throws ArgumentError state["02"]

        @test dot(state, state) ≈ 1
        @test norm(state) ≈ 1

        state_first = apply(X(2), SparseState{K,V}(2))
        state_second = apply(X(3), SparseState{K,V}(3))
        @test state_first["01"] ≈ 1
        @test state_second["001"] ≈ 1
        @test norm(state_first) ≈ norm(state_second) ≈ 1
        state_prod = kron(state_first, state_second)
        @test num_qubits(state_prod) == 2 + 3
        @test length(state_prod) == 1
        @test state_prod["01001"] ≈ 1
    end
end

@testset "Circuits and operators" begin
    # TODO
end

@testset "Single-qubit superposition" begin
    state_initial = SparseState(1)
    circuit = H(1)
    state_final = circuit(state_initial)
    @test haskey(state_final, "0") && haskey(state_final, "1")
    @test state_final["0"] ≈ state_final["1"] ≈ 1 / sqrt(2)

    shots = 10000
    @test expectation(state_initial, 1) ≈ 0
    @test expectation(state_final, 1) ≈ 1 / 2
    outcomes = [expectation(apply(Measure(1), state_final), 1) for _ in 1:shots]
    @test ≈(sum(outcomes) / shots, 1 / 2; atol=(4 / √shots))
end

@testset "Bell state" begin
    state_initial = SparseState(2)
    circuit = H(1) * CX(1, 2)
    state_final = circuit(state_initial)
    @test haskey(state_final, "00") && haskey(state_final, "11")
    @test state_final["00"] ≈ state_final["11"] ≈ 1 / sqrt(2)

    shots = 10000
    @test expectation(state_final, 1) ≈ 1 / 2
    @test expectation(state_final, 2) ≈ 1 / 2
    outcomes = [expectation(apply(Measure([1, 2]), state_final), [1, 2]) for _ in 1:shots]
    @test all(isapprox(rec...; atol=4 * eps()) for rec in outcomes)
end

@testset "Bacon-Shor error-correction" begin
    # Circuit adapted from https://arxiv.org/abs/2404.11663
    p = 0.0082 # pseudo-threshold for `Z` eigenstates
    shots = 100000

    circuit_encoding = Circuit(
        H(1), CX(1, 4), CX(4, 7), H([1, 4, 7]), CX([(1, 2), (4, 5), (7, 8)]), CX([(2, 3), (5, 6), (8, 9)])
    )

    state_initial = SparseState{UInt16}(12)
    state_encoded = circuit_encoding(state_initial)
    @test length(state_encoded) == 4

    circuit_correction_x = Circuit(
        H([10, 11, 12]),
        CX([(10, 1), (11, 4), (12, 7)]),
        CX([(10, 4), (11, 7), (12, 1)]),
        CX([(10, 2), (11, 5), (12, 8)]),
        CX([(10, 5), (11, 8), (12, 2)]),
        CX([(10, 3), (11, 6), (12, 9)]),
        CX([(10, 6), (11, 9), (12, 3)]),
        H([10, 11, 12]),
        CCZ(10, 12, 1),
        CCZ(10, 11, 4),
        CCZ(11, 12, 7),
        Reset([10, 11, 12]),
    )

    circuit_correction_z = Circuit(
        CX([(1, 10), (2, 11), (3, 12)]),
        CX([(2, 10), (3, 11), (1, 12)]),
        CX([(4, 10), (5, 11), (6, 12)]),
        CX([(5, 10), (6, 11), (4, 12)]),
        CX([(7, 10), (8, 11), (9, 12)]),
        CX([(8, 10), (9, 11), (7, 12)]),
        CCX(10, 12, 1),
        CCX(10, 11, 2),
        CCX(11, 12, 3),
        Reset([10, 11, 12]),
    )
    noise_model = [
        H => DepolarizingChannel{1}(p),
        X => DepolarizingChannel{1}(p),
        Z => DepolarizingChannel{1}(p),
        CX => DepolarizingChannel{2}(p),
        CZ => DepolarizingChannel{2}(p),
        CCX => DepolarizingChannel{3}(p),
        CCZ => DepolarizingChannel{3}(p),
    ]

    circuit_noisy = Circuit(
        append_operators(circuit_correction_x, noise_model...),
        append_operators(circuit_correction_z, noise_model...),
        circuit_correction_x,
        circuit_correction_z,
    )

    stabilizer_indices = Dict(
        X => [[1, 2, 3, 4, 5, 6], [4, 5, 6, 7, 8, 9], [7, 8, 9, 1, 2, 3]],
        Z => [[1, 2, 4, 5, 7, 8], [2, 3, 5, 6, 8, 9], [3, 1, 6, 4, 9, 7]],
    )

    stabilizers = Dict(
        X => [Circuit(X(i) for i in inds) for inds in stabilizer_indices[X]],
        Z => [Circuit(Z(i) for i in inds) for inds in stabilizer_indices[Z]],
    )
    operator_logical = Z(i for i in 1:9)

    state_corrected = circuit_noisy(state_encoded)
    @test all(dot(state_corrected, S, state_corrected) ≈ 1.0 for S in stabilizers[X])
    @test all(dot(state_corrected, S, state_corrected) ≈ 1.0 for S in stabilizers[Z])
    @test norm(state_corrected) ≈ 1.0
    @test real(dot(state_corrected, operator_logical, state_corrected)) ≈ 1.0

    logical_error_rate = 0.0
    for _ in 1:shots
        state_corrected = circuit_noisy(state_encoded)
        logical_error_rate +=
            1 - isapprox(real(dot(state_corrected, operator_logical, state_corrected)), 1.0; atol=1e-12)
    end
    logical_error_rate /= shots
    @test ≈(logical_error_rate, p; rtol=0.25)
end
