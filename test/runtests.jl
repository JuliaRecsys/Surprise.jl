using Test

using Persa
using Surprise
using DatasetsCF

dataset = DatasetsCF.MovieLens()

@testset "Memory Based Algorithms" begin
    model = Surprise.KNNBasic(dataset)
    Persa.train!(model, dataset)
    @test model[1,1] >= 0

    model = Surprise.KNNBaseline(dataset)
    Persa.train!(model, dataset)
    @test model[1,1] >= 0

    model = Surprise.KNNWithMeans(dataset)
    Persa.train!(model, dataset)
    @test model[1,1] >= 0

    model = Surprise.SlopeOne(dataset)
    Persa.train!(model, dataset)
    @test model[1,1] >= 0
end
