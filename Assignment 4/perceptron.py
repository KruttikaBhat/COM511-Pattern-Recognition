import numpy as np
import matplotlib.pyplot as plt


a=[0,0,0]

initial_data=[[0.5,3,2],
              [1.0,3.0,2],
              [0.5,2.5,2],
              [1.0,2.5,2],
              [1.5,2.5,2],
              [4.5,1.0,1],
              [5.0,1.0,1],
              [4.5,0.5,1],
              [5.5,0.5,1]]
"""
initial_data=[[0,0,1], #class 1-> output is 0, class 2-> output is 1
              [0,1,2],
              [1,0,2],
              [1,1,2]]
"""
data=[]
for i in range(len(initial_data)):
    if initial_data[i][len(initial_data[0])-1]==2:
        data.append(-np.array(initial_data[i]))
        data[i][len(initial_data[0])-1]=-1
    else:
        data.append(initial_data[i])
print(data)

d=len(data)
c=0
i=0
lr=1

while c!=d:
    check=np.dot(np.array(data[i]),np.array(a).T)
    #print(check)
    #print(i)
    if check<=0:
        for j in range(len(a)):
            a[j]=a[j]+(lr*data[i][j])
        print(a)
        c=0
    else:
        c=c+1
    i=(i+1)%d

print("a=",a)


fig = plt.figure()
ax = fig.gca()
ax.set_xticks(np.arange(-10, 10, 1))
ax.set_yticks(np.arange(-10, 10, 1))


xcoor=[]
ycoor=[]
x = np.arange(-10, 10, 1)
a2=a[0]
a1=a[1]
a0=a[2]

for i in x:
    #a2*x1 + a1*x2 +a0=0
    y1=(-a0-(a2*i))/a1
    xcoor.append(i)
    ycoor.append(y1)

#print(a2,a1,a0)
#print(xcoor)
#print(ycoor)


plt.plot(xcoor, ycoor,color='blue')

for i in range(d):
    if initial_data[i][len(initial_data[0])-1]==2:
        class2=plt.scatter(initial_data[i][0], initial_data[i][1], color="green",marker='o')
        class2negate=plt.scatter(data[i][0], data[i][1], color="purple",marker='o')
    else:
        class1=plt.scatter(initial_data[i][0], initial_data[i][1], color="red",marker='*')

plt.legend((class1, class2, class2negate),
           ('class 1', 'class 2', 'class 2 after negation'),
           loc='upper left',
           scatterpoints=1,
           fontsize=8)

plt.title('Perceptron Classification')
plt.xlabel("x1")
plt.ylabel("x2")

plt.grid()



plt.show()
