# Surprise.jl - Wrapper to Surprise Python Package

[![Build Status](https://travis-ci.org/JuliaRecsys/Surprise.jl.svg?branch=master)](https://travis-ci.org/JuliaRecsys/Surprise.jl)
[![Coverage Status](https://coveralls.io/repos/JuliaRecsys/Surprise.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/JuliaRecsys/Surprise.jl?branch=master)
[![codecov.io](http://codecov.io/github/JuliaRecsys/Surprise.jl/coverage.svg?branch=master)](http://codecov.io/github/JuliaRecsys/Surprise.jl?branch=master)

**Installation**: at the Julia REPL, `Pkg.add("Surprise")`

**Reporting Issues and Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)

## Goal

The main aim is to create a framework that facilitates the study of recommender systems in Julia.

## Example

```
julia> using Persa

julia> using DatasetsCF

julia> using Surprise

julia> dataset = DatasetsCF.MovieLens();

julia> (ds_train, ds_test) = Persa.get(Persa.HoldOut(dataset, 0.9));

julia> model = Surprise.IRSVD(ds_train);

julia> Persa.train!(model, ds_train)

julia> print(Persa.aval(model, ds_test))
MAE - 0.7380270708513149
RMSE - 0.9369961685890544
Coverage - 0.9988001199880012
```

## Algorithms

List of collaborative algorithms:

Algorithm      | Description
-------------|------------------------------------------------------------------------
KNNBasic  | A basic KNN algorithm.
KNNBaseline    | A basic KNN algorithm but using a baseline factor.
KNNWithMeans    | A basic KNN algorithm but using user or item mean.
SlopeOne    | SlopeOne algorithm.
RSVD    | Regulared SVD. The algorithm is also known as SVD by Funk.
IRSVD    | Improved Regulared SVD. Extension of RSVD algorithm adding the user and item bias.
