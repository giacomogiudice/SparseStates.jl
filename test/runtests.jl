using Test
using LinearAlgebra
using SparseStates

const KEY_TYPES = (UInt8, UInt16, UInt32, UInt64, UInt128)
const REAL_VAL_TYPES = (Float32, Float64)
const COMPLEX_VAL_TYPES = (ComplexF32, ComplexF64)
const VAL_TYPES = (REAL_VAL_TYPES..., COMPLEX_VAL_TYPES...)

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

        state_first = SparseState{K,V}("01" => 1, 2)
        state_second = SparseState{K,V}("001" => 1, 3)
        @test state_first["01"] ≈ 1
        @test state_second["001"] ≈ 1
        @test norm(state_first) ≈ norm(state_second) ≈ 1
        state_prod = kron(state_first, state_second)
        @test num_qubits(state_prod) == 2 + 3
        @test length(state_prod) == 1
        @test state_prod["01001"] ≈ 1

        state = SparseState{K,V}(["00" => 1, "01" => 1, "10" => 1, "11" => 1], 2)
        @test norm(state) ≈ 2
        normalize!(state)
        @test norm(state) ≈ 1
    end
end

@testset "Single-qubit operators" begin
    @testset "Single-qubit operator for SparseState{$K,$V}" for (K, V) in
                                                                Iterators.product(KEY_TYPES, COMPLEX_VAL_TYPES)
        N = rand(1:(8 * sizeof(K) - 1))
        n = rand(1:N)
        arr = rand(0:1, N)
        arr[n] = 0
        up = SparseState{K,V}(join(arr) => 1, N)
        arr[n] = 1
        dn = SparseState{K,V}(join(arr) => 1, N)

        α, β = normalize!(randn(V, 2))
        state_initial = α * up + β * dn
        @test norm(state_initial) ≈ 1

        # `X` gate
        state_final = @inferred apply(X(n), state_initial)
        @test state_final ≈ β * up + α * dn

        # `Y` gate
        state_final = @inferred apply(Y(n), state_initial)
        @test state_final ≈ -im * β * up + im * α * dn

        # `Z` gate
        state_final = @inferred apply(Z(n), state_initial)
        @test state_final ≈ α * up - β * dn

        # `H` gate
        state_final = @inferred apply(H(n), state_initial)
        @test state_final ≈ (α + β) / √2 * up + (α - β) / √2 * dn

        # `S` gate
        state_final = @inferred apply(S(n), state_initial)
        @test state_final ≈ α * up + im * β * dn

        # `T` gate
        state_final = @inferred apply(T(n), state_initial)
        @test state_final ≈ α * up + √im * β * dn

        # `U` gate
        θ, ϕ, λ = 4π * rand(real(V), 3)
        state_final = @inferred apply(U(n; θ, ϕ, λ), state_initial)
        @test state_final ≈
            (α * exp(-(im / 2) * (ϕ + λ)) * cos(θ / 2) - β * exp(-(im / 2) * (ϕ - λ)) * sin(θ / 2)) * up +
              (α * exp(+(im / 2) * (ϕ - λ)) * sin(θ / 2) + β * exp(+(im / 2) * (ϕ + λ)) * cos(θ / 2)) * dn

        # `RX` gate
        θ = 4π * rand(real(V))
        state_final = @inferred apply(RX(n; θ), state_initial)
        @test state_final ≈ (α * cos(θ / 2) - im * β * sin(θ / 2)) * up + (-im * α * sin(θ / 2) + β * cos(θ / 2)) * dn

        # `RY` gate
        θ = 4π * rand(real(V))
        state_final = @inferred apply(RY(n; θ), state_initial)
        @test state_final ≈ (α * cos(θ / 2) - β * sin(θ / 2)) * up + (α * sin(θ / 2) + β * cos(θ / 2)) * dn

        # `RZ` gate
        θ = 4π * rand(real(V))
        state_final = @inferred apply(RZ(n; θ), state_initial)
        @test state_final ≈ α * exp(-(im / 2) * θ) * up + β * exp(+(im / 2) * θ) * dn
    end
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

@testset "Bacon-Shor error correction" begin
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
