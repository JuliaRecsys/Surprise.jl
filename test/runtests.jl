using Surprise
reload("Surprise")
using Base.Test
using Persa
reload("Persa")
#Testing
cf = Persa
ds = cf.createdummydataset()

holdout = cf.HoldOut(ds, 0.9)

(ds_train, ds_test) = cf.get(holdout)
#####
model = Surprise.SurpriseKNNWithMeans(ds_train; k = 60, min_k = 60)
####
cf.train!(model, ds_train)

@assert cf.aval(model, ds_test).mae >= 0.0

typeof(ds_train)


#######
a = Surprise.GlobalMean3()
