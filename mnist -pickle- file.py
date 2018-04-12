import numpy as np

"""#python2   --its working
import gzip, pickle

with gzip.open('mnist.pkl.gz','rb') as ff :
    train, val, test = list(pickle.load( ff ))

print(len(train))

print(train[0].shape)

print(train)"""







import gzip,pickle
with gzip.open('mnist.pkl.gz','rb') as ff:
    u=pickle._Unpickler(ff)

    u=encoding='latin1'
    train,val,tet=u.load(ff)


print(train)

print(train[0].shape)
print(train[1])
print(len(train))
print(val)
print(val[1])
print(val[0])
print(len(val))

if len(train)==5:
    print (True)
else:
    print(False)

elements=train[0]
print(elements[1:5])

print(elements[2])
