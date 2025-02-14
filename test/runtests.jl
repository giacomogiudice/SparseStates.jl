using Test
using Random
using LinearAlgebra
using SparseStates

# Set randomness seed for reproducibility 
Random.seed!(42)

const KEY_TYPES = (UInt8, UInt16, UInt32, UInt64, UInt128)
const REAL_VAL_TYPES = (Float32, Float64)
const COMPLEX_VAL_TYPES = (ComplexF32, ComplexF64)
const VAL_TYPES = (REAL_VAL_TYPES..., COMPLEX_VAL_TYPES...)

@testset "SparseStates" begin
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

@testset "Operators" begin
    @testset "Single-qubit gates with SparseState{$K,$V}" for (K, V) in Iterators.product(KEY_TYPES, COMPLEX_VAL_TYPES)
        N = rand(1:(8 * sizeof(K) - 1))
        n = rand(1:N)
        arr = rand(0:1, N)
        c = Dict{String,V}()
        basis = Dict{String,SparseState{K,V}}()
        arr[n] = 0
        basis["0"] = SparseState{K,V}(join(arr) => 1, N)
        arr[n] = 1
        basis["1"] = SparseState{K,V}(join(arr) => 1, N)

        # Create a superposition of basis states
        α, β = normalize!(randn(V, 2))
        state_initial = α * basis["0"] + β * basis["1"]
        @test norm(state_initial) ≈ 1

        # `X` gate
        state_final = @inferred apply(X(n), state_initial)
        @test state_final ≈ β * basis["0"] + α * basis["1"]

        # `Y` gate
        state_final = @inferred apply(Y(n), state_initial)
        @test state_final ≈ -im * β * basis["0"] + im * α * basis["1"]

        # `Z` gate
        state_final = @inferred apply(Z(n), state_initial)
        @test state_final ≈ α * basis["0"] - β * basis["1"]

        # `H` gate
        state_final = @inferred apply(H(n), state_initial)
        @test state_final ≈ (α + β) / √2 * basis["0"] + (α - β) / √2 * basis["1"]

        # `S` gate
        state_final = @inferred apply(S(n), state_initial)
        @test state_final ≈ α * basis["0"] + im * β * basis["1"]

        # `T` gate
        state_final = @inferred apply(T(n), state_initial)
        @test state_final ≈ α * basis["0"] + √im * β * basis["1"]

        # `U` gate
        θ, ϕ, λ = 4π * rand(real(V), 3)
        state_final = @inferred apply(U(n; θ, ϕ, λ), state_initial)
        @test state_final ≈
            (α * exp(-(im / 2) * (ϕ + λ)) * cos(θ / 2) - β * exp(-(im / 2) * (ϕ - λ)) * sin(θ / 2)) * basis["0"] +
              (α * exp(+(im / 2) * (ϕ - λ)) * sin(θ / 2) + β * exp(+(im / 2) * (ϕ + λ)) * cos(θ / 2)) * basis["1"]

        # `RX` gate
        θ = 4π * rand(real(V))
        state_final = @inferred apply(RX(n; θ), state_initial)
        @test state_final ≈
            (α * cos(θ / 2) - im * β * sin(θ / 2)) * basis["0"] + (-im * α * sin(θ / 2) + β * cos(θ / 2)) * basis["1"]

        # `RY` gate
        θ = 4π * rand(real(V))
        state_final = @inferred apply(RY(n; θ), state_initial)
        @test state_final ≈
            (α * cos(θ / 2) - β * sin(θ / 2)) * basis["0"] + (α * sin(θ / 2) + β * cos(θ / 2)) * basis["1"]

        # `RZ` gate
        θ = 4π * rand(real(V))
        state_final = @inferred apply(RZ(n; θ), state_initial)
        @test state_final ≈ α * exp(-(im / 2) * θ) * basis["0"] + β * exp(+(im / 2) * θ) * basis["1"]
    end

    @testset "Two-qubit gates with SparseState{$K,$V}" for (K, V) in Iterators.product(KEY_TYPES, COMPLEX_VAL_TYPES)
        N = rand(1:(8 * sizeof(K) - 1))
        m, n = randperm(N)[1:2]
        arr = rand(0:1, N)

        c = Dict{String,V}()
        basis = Dict{String,SparseState{K,V}}()
        coeffs = normalize!(randn(V, 2^2))
        for (comb, v) in zip(Iterators.product(0:1, 0:1), coeffs)
            arr[[m, n]] .= comb
            basis[join(comb)] = SparseState{K,V}(join(arr) => 1, N)
            c[join(comb)] = v
        end

        # Create a superposition of basis states
        state_initial = c["00"] * basis["00"] + c["01"] * basis["01"] + c["10"] * basis["10"] + c["11"] * basis["11"]
        @test norm(state_initial) ≈ 1

        # `CX` gate
        state_final = @inferred apply(CX(m, n), state_initial)
        state_test = c["00"] * basis["00"] + c["01"] * basis["01"] + c["11"] * basis["10"] + c["10"] * basis["11"]
        @test state_final ≈ state_test

        # `CY` gate
        state_final = @inferred apply(CY(m, n), state_initial)
        state_test =
            c["00"] * basis["00"] + c["01"] * basis["01"] - im * c["11"] * basis["10"] + im * c["10"] * basis["11"]
        @test state_final ≈ state_test

        # `CZ` gate
        state_final = @inferred apply(CZ(m, n), state_initial)
        state_test = c["00"] * basis["00"] + c["01"] * basis["01"] + c["10"] * basis["10"] - c["11"] * basis["11"]

        # `SWAP` gate
        state_final = @inferred apply(SWAP(m, n), state_initial)
        state_test = c["00"] * basis["00"] + c["10"] * basis["01"] + c["01"] * basis["10"] + c["11"] * basis["11"]
    end

    @testset "Three-qubit gates with SparseState{$K,$V}" for (K, V) in Iterators.product(KEY_TYPES, COMPLEX_VAL_TYPES)
        N = rand(1:(8 * sizeof(K) - 1))
        m, n, l = randperm(N)[1:3]
        arr = rand(0:1, N)

        c = Dict{String,V}()
        basis = Dict{String,SparseState{K,V}}()
        coeffs = normalize!(randn(V, 2^3))
        for (comb, v) in zip(Iterators.product(0:1, 0:1, 0:1), coeffs)
            arr[[m, n, l]] .= comb
            basis[join(comb)] = SparseState{K,V}(join(arr) => 1, N)
            c[join(comb)] = v
        end

        # Create a superposition of basis states
        state_initial =
            c["000"] * basis["000"] +
            c["001"] * basis["001"] +
            c["010"] * basis["010"] +
            c["011"] * basis["011"] +
            c["100"] * basis["100"] +
            c["101"] * basis["101"] +
            c["110"] * basis["110"] +
            c["111"] * basis["111"]
        @test norm(state_initial) ≈ 1

        # `CCX` gate
        state_final = @inferred apply(CCX(m, n, l), state_initial)
        state_test =
            c["000"] * basis["000"] +
            c["001"] * basis["001"] +
            c["010"] * basis["010"] +
            c["011"] * basis["011"] +
            c["100"] * basis["100"] +
            c["101"] * basis["101"] +
            c["111"] * basis["110"] +
            c["110"] * basis["111"]
        @test state_final ≈ state_test

        # `CCY` gate
        state_final = @inferred apply(CCY(m, n, l), state_initial)
        state_test =
            c["000"] * basis["000"] +
            c["001"] * basis["001"] +
            c["010"] * basis["010"] +
            c["011"] * basis["011"] +
            c["100"] * basis["100"] +
            c["101"] * basis["101"] +
            -im * c["111"] * basis["110"] +
            +im * c["110"] * basis["111"]
        @test state_final ≈ state_test

        # `CCZ` gate
        state_final = @inferred apply(CCZ(m, n, l), state_initial)
        state_test =
            c["000"] * basis["000"] +
            c["001"] * basis["001"] +
            c["010"] * basis["010"] +
            c["011"] * basis["011"] +
            c["100"] * basis["100"] +
            c["101"] * basis["101"] +
            c["110"] * basis["110"] +
            -c["111"] * basis["111"]
        @test state_final ≈ state_test
    end
end

@testset "Circuits" begin
    circuit = H(1) * CX(1, 2)
    @test circuit isa Circuit
    @test circuit * CCX(1, 2, 3) isa Circuit
    @test CCX(1, 2, 3) * circuit isa Circuit
end

@testset "Utilities" begin
    state = SparseState(("00" => 1, "11" => 1), 2)
    normalize!(state)
    observables = pauli_decomposition(state)
    @test sum(weight * real(dot(state, op, state)) for (op, weight) in observables) ≈ 1

    state = SparseState(("000" => randn(), "111" => randn()), 3)
    normalize!(state)
    observables = pauli_decomposition(state)
    @test sum(weight * real(dot(state, op, state)) for (op, weight) in observables) ≈ 1
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
    outcomes = Bool[]
    for _ in 1:shots
        measurement = Measure(1; callback=(out, _) -> push!(outcomes, only(out)))
        apply(measurement, state_final)
    end
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

    outcomes = Bool[]
    for _ in 1:shots
        measurement = Measure(1; callback=(out, _) -> push!(outcomes, allequal(out)))
        apply(measurement, state_final)
    end
    @test all(outcomes)
end

@testset "Teleportation" begin
    circuit_teleportation = Circuit(
        H(2),
        CX(2, 3),
        CX(1, 2),
        MeasureOperator(Z(2); callback=((rec, state) -> only(rec) && apply!(X(3), state))),
        Reset(2),
        MeasureOperator(X(1); callback=((rec, state) -> only(rec) && apply!(Z(3), state))),
        Reset(1),
        SWAP(3, 1),
    )

    state_zero = SparseState{UInt8}(3)
    state_one = apply(X(1), state_zero)

    for _ in 1:10
        α, β = normalize!(randn(ComplexF64, 2))

        state_initial = α * state_zero + β * state_one
        state_final = apply(circuit_teleportation, state_initial)

        @test abs2(dot(state_zero, state_final)) ≈ abs2(α)
        @test abs2(dot(state_one, state_final)) ≈ abs2(β)
        @test abs2(dot(state_final, state_initial)) ≈ 1
    end
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
        H => DepolarizingChannel{1}(; p),
        X => DepolarizingChannel{1}(; p),
        Z => DepolarizingChannel{1}(; p),
        CX => DepolarizingChannel{2}(; p),
        CZ => DepolarizingChannel{2}(; p),
        CCX => DepolarizingChannel{3}(; p),
        CCZ => DepolarizingChannel{3}(; p),
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
