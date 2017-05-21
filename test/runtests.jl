using Base.Test

using Surprise
using Persa

#Testing
ds = Persa.createdummydataset()

holdout = Persa.HoldOut(ds, 0.9)

(ds_train, ds_test) = Persa.get(holdout)

model = Surprise.IRSVD(ds_train)

Persa.train!(model, ds_train)

@assert Persa.aval(model, ds_test).mae >= 0.0
