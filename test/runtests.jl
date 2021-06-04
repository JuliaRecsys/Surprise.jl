using Test

using Persa
using Surprise
using DatasetsCF

dataset = DatasetsCF.MovieLens()

model = Surprise.KNNBasic(dataset)

Persa.train!(model, dataset)

@test model[1,1] >= 0
