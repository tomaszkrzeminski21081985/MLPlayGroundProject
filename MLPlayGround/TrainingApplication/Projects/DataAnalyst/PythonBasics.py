import datetime
import numpy as np


data=[np.random.standard_normal() for i in range(10)]


print(data,sep='\n')


for i in range(10):
    if(i<5):
        print("smaller then five")
    else:
     print("bigger then 5") 
              



a="test"
print(a)
b=a;
print(b)

a=a+"bbbb"

print(a,b)


c=[1,2,3]

d=c

d.append(5)


print(d)
print(c)


e=5

f=e

e=e+1

print(e,f)




class Person:

 def addToTable(self,element,add):
        element.append(add) 
        return element
        

     

person= Person()
y=[1,2,3,4]
x=  person.addToTable(y,5)
print(x,y)



a="test"

k=(3,5)

print(k[1])


l=(1,2 ,(4,6))

print(l[2])

x=(1,2,([1,2,3]),4)

x[2].append(10)

print(x)

k=((1,2,3),(4,5,6),(7,8,9))

for a,b,c in k: 
    print(a , b , c)


    a,*_=k
    print(a)
    print(*_)


l1=list(range(0,10))


print(l1)

print([l1[0:1]])

l1[1:3]=[11,12,13]

print(l1)


l3=[ abs(i) for i in range(-100,20) if i<0  ]

print(l3)


k1=((1,2,3),(4,5,6),(7,8,9))

for i in k1:
    print(i)



flatten=[ l  for k in k1 for l in k     ]


# print(flatten)


# t111=((x, y) for x in range(2) for y in range(x))


# for z in t111:
#     print(z)
    



# arr=np.array([[1,2,3],[4,5,6]])

# print(arr)

# arr1=arr*10

# print(arr1)


# print(arr*arr)


arr2=np.ones((3,3,3))


arr2[1]*=  2
arr2[2]*= 3

print(arr2)


test4=np.array([["Tomek"],["Ada"],["Tomek"],["Ada"],["Tomek"],["Ada"]])


print(test4.shape)


names = np.array(["Bob", "Joe", "Will", "Bob", "Will", "Joe", "Joe"])
data = np.array([[4, 7], [0, 2], [-5, 6], [0, 0], [1, 2],
                 [-12, -4], [3, 4]])




# data[data < 0] = 0

# print(data)


# test666=names[names=="Bob"]="Tomek"


# print(names)
# print(test666)



# test123=np.arange(10)

# print(test123)


# xxx=np.add(test123,100)

# print(test123)

# print(xxx)



table1=np.array(range(16))

table1=table1.reshape(4,4)

table1[ :,  0]=3

print(table1)

table1.sort(axis=0)

print(table1)

table1.sort(axis=1)

print(table1)