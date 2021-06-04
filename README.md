# Surprise.jl - Wrapper to Surprise Python Package

[![][ci-img]][ci-url]
[![][codecov-img]][codecov-url]

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

[ci-img]: https://img.shields.io/github/checks-status/JuliaRecsys/Surprise.jl/master?style=flat-square
[ci-url]: https://github.com/JuliaRecsys/Surprise.jl/actions

[codecov-img]: https://img.shields.io/codecov/c/github/JuliaRecsys/Surprise.jl?style=flat-square
[codecov-url]: https://codecov.io/gh/JuliaRecsys/Surprise.jl