using Test
using SparseStates

@testset "SparseState initialization" begin
    state = SparseState{UInt8,Float64}(2)
    @test state isa AbstractDict
    @test (@inferred keytype(state)) == UInt8 && (@inferred valtype(state)) == Float64
    @test (@inferred num_qubits(state)) == 2
    @test (@inferred length(state)) == 1
    @test (@inferred haskey(state, 0))
    @test (@inferred haskey(state, "00"))
    @test (@inferred state[0]) ≈ 1
    @test (@inferred state["00"]) ≈ 1
end

@testset "SparseState operations" begin
    state_first = apply(X(2), SparseState{UInt16,Float64}(2))
    state_second = apply(X(3), SparseState{UInt16,Float64}(3))
    @test state_first["01"] ≈ 1
    @test state_second["001"] ≈ 1
    state_prod = kron(state_first, state_second)
    @test num_qubits(state_prod) == 2 + 3
    @test length(state_prod) == 1
    @test state_prod["01001"] ≈ 1
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
