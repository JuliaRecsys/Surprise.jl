using Persa
using Surprise

ds = Persa.createdummydataset()

for (ds_train, ds_test) in cf.KFolds(ds, 10)
  model = Surprise.SurpriseIRSVD(ds_train)

  Persa.train!(model, ds_train)

  print(Persa.aval(model, ds_test, Persa.recommendation(ds)))
end
