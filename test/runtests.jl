using Test
using SparseStates

@testset "SparseState initialization" begin
    state = SparseState{UInt8,Float64}(2)
    @test state isa AbstractDict
    @test (@inferred keytype(state)) == UInt8 && (@inferred valtype(state)) == Float64
end
