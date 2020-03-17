import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns


#设定超参数
#N = 70             #申请用户数量
#M = 8              #可用基站数
#K = 100              #可用基站频点

EPXILONG = 0.2      #设置ε值

def Location_matrix_df(n,m,k):
    Location_matrix = np.zeros(shape=(n,m,k),dtype=int)
    for i in range (n):
        Location_matrix[i,int(m * random.random()),0:k] = 1
    return Location_matrix

def Location_matrix_show(Location_matrix):
    Location_matrix_2D = Location_matrix[:,:,0]
    sns.heatmap(Location_matrix_2D ,annot=False, vmin=0, vmax=1, cmap="Blues", xticklabels=False  ,
                    yticklabels =False   )
    plt.xlabel ('Base')
    plt.ylabel ('Users')
    plt.show()

def Allocation_matrix_show(n,m,k,Allocation_matrix):
    Allocation_matrix_2D = Allocation_matrix.reshape(n,m * k)
    sns.heatmap(Allocation_matrix_2D ,annot=False, vmin=0, vmax=1, cmap="Greens", xticklabels=False,
                    yticklabels =False   )
    plt.xlabel ('Base * Frequence')
    plt.ylabel ('Users')
    plt.show()

#定义随机分配矩阵
def Random_Allocation_matrix_df(n,m,k,Location_matrix):
    Random_Allocation_matrix = np.zeros(shape=(n,m,k),dtype=int)
    num = 0
    flag = 0
    for i in range(n):
        for l in range(m):
            if np.all(Location_matrix[i, l, 0]) == 1:
                for x in range(k):
                    for j in range(n):
                        flag = flag + Random_Allocation_matrix[j, l, x]  # 观察x号频段是否有人使用

                    if flag == 1:
                        flag = 0
                        #print("对于基站%d范围内%d号用户,频段 %d 已被占用" % (l,i,x))
                    else:
                        flag = 0
                        Random_Allocation_matrix[i, l, x] = 1
                        #print("基站%d范围内%d号用户,频段 %d 成功分配" % (l, i, x))
                        #print("%d号用户,随机分配成功" % ( i))
                        num = num + 1
                        break
        #if np.sum(Random_Allocation_matrix[i,:,:])==0:
            #print("%d号用户,随机分配失败" % ( i))

    return Random_Allocation_matrix,num

#定义贪心算法分配矩阵
def Greedy_Allocation_matrix(n,m,k,Location_matrix):
    Greedy_Allocation_matrix = np.zeros(shape=(n, m, k), dtype=int)
    Greedy_Allocation_matrix_2 = np.zeros(shape=(n, m, k), dtype=int)
    I_matrix = np.zeros(shape=(n, m, k), dtype=int)
    flag = 0
    max_R = 0
    next_R = 0
    num = 0
    for i in range (n):
        for l in range(m):
            if np.all(Location_matrix[i, l, 0]) == 1:
                max_R = 0
                next_R = 0
                Greedy_Allocation_matrix_2[i,l,:] = Greedy_Allocation_matrix[i,l,:]
                for x in range(k):
                    for j in range(n):
                        flag = flag + Greedy_Allocation_matrix[j, l, x]  # 观察x号频段是否有人使用

                    if flag == 1:
                        flag = 0
                        #print("对于基站%d范围内%d号用户,频段 %d 已被占用" % (l, i, x))

                    else:
                        flag = 0
                        Greedy_Allocation_matrix_2[i, l, x] = 1
                        I_matrix[:,:,x] = I_matrix[:,:,x] +1
                        I_matrix[:, l, x] = I_matrix[:, l, x] -1
                        next_R = R_caculate(n,m,k,Greedy_Allocation_matrix_2,I_matrix)
                        Greedy_Allocation_matrix_2[i, l, x] = 0
                        I_matrix[:, :, x] = I_matrix[:, :, x] -1
                        I_matrix[:, l, x] = I_matrix[:, l, x] +1
                        if (next_R > max_R):
                            Greedy_Allocation_matrix[i, l, :] = 0
                            Greedy_Allocation_matrix[i, l, x] = 1
                            max_R = next_R
                            I_matrix[:, :, x] = I_matrix[:, :, x] + 1
                            I_matrix[:, l, x] = I_matrix[:, l, x] - 1
                            #print("贪心算法更迭一次")
        if np.sum(Greedy_Allocation_matrix[i, :, :]) == 0:
             pass#print("%d号用户,贪心分配失败" % (i))
        else:
            #print("%d号用户,贪心分配成功" % (i))
            num = num +1
    return Greedy_Allocation_matrix,num

#定义ε—贪心算法分配矩阵
def Epxilong_Greedy_Allocation_matrix(n,m,k,Location_matrix):
    Ep_Greedy_Allocation_matrix = np.zeros(shape=(n, m, k), dtype=int)
    Ep_Greedy_Allocation_matrix_2 = np.zeros(shape=(n, m, k), dtype=int)
    I_matrix = np.zeros(shape=(n, m, k), dtype=int)
    flag = 0
    max_R = 0
    next_R = 0
    num = 0
    epsilong = 0
    for i in range (n):
        epsilong = random.random()
        if epsilong >=EPXILONG :
            #print("%d号申请者使用贪心分配" %(i))
            for l in range(m):
                if np.all(Location_matrix[i, l, 0]) == 1:
                    max_R = 0
                    next_R = 0
                    Ep_Greedy_Allocation_matrix_2[i,l,:] = Ep_Greedy_Allocation_matrix[i,l,:]
                    for x in range(k):
                        for j in range(n):
                            flag = flag + Ep_Greedy_Allocation_matrix[j, l, x]  # 观察x号频段是否有人使用

                        if flag == 1:
                            flag = 0
                            #print("对于基站%d范围内%d号用户,频段 %d 已被占用" % (l, i, x))

                        else:
                            flag = 0
                            Ep_Greedy_Allocation_matrix_2[i, l, x] = 1
                            I_matrix[:, :, x] = I_matrix[:, :, x] + 1
                            I_matrix[:, l, x] = I_matrix[:, l, x] - 1
                            next_R = R_caculate(n,m,k,Ep_Greedy_Allocation_matrix_2,I_matrix)
                            Ep_Greedy_Allocation_matrix_2[i, l, x] = 0
                            I_matrix[:, :, x] = I_matrix[:, :, x] - 1
                            I_matrix[:, l, x] = I_matrix[:, l, x] + 1
                            if (next_R > max_R):
                                Ep_Greedy_Allocation_matrix[i, l, :] = 0
                                Ep_Greedy_Allocation_matrix[i, l, x] = 1
                                max_R = next_R
                                I_matrix[:, :, x] = I_matrix[:, :, x] + 1
                                I_matrix[:, l, x] = I_matrix[:, l, x] - 1
                                #print("贪心算法更迭一次")
        else:
            #print("%d号申请者使用随机分配" % (i))
            for l in range(m):
                if np.all(Location_matrix[i, l, 0]) == 1:
                    for x in range(k):
                        for j in range(n):
                            flag = flag + Ep_Greedy_Allocation_matrix[j, l, x]  # 观察x号频段是否有人使用

                        if flag == 1:
                            flag = 0
                            #print("对于基站%d范围内%d号用户,频段 %d 已被占用" % (l, i, x))
                        else:
                            flag = 0
                            Ep_Greedy_Allocation_matrix[i, l, x] = 1
                            #print("基站%d范围内%d号用户,频段 %d 成功分配" % (l, i, x))
                            #print("%d号用户,随机分配成功" % (i))
                            break


        if np.sum(Ep_Greedy_Allocation_matrix[i, :, :]) == 0:
            pass#print("%d号用户,随机分配失败" % (i))
        else:
            #print("%d号用户,随机分配成功" % (i))
            num = num +1
    return Ep_Greedy_Allocation_matrix,num

#计算分配矩阵干扰
def I_caculate(n,m,k,Allocation_matrix):
    I_matrix = np.zeros(shape=(n,m,k),dtype=int)
    for l in range (k):
        for i in range (n):
            for j in range (m):
                I_matrix[i,j,l] = np.sum(Allocation_matrix[:,:,l]) - np.sum(Allocation_matrix[:,j,l])
    return I_matrix

#计算分配矩阵传输数据量
def R_caculate(n,m,k,Allocation_matrix,I_matrix):
    Allocation_matrix_float = Allocation_matrix.astype(np.float)
    r = np.sum(np.log2(1 + Allocation_matrix_float/(I_matrix/(n/(m*k)) + 0.01)))
    return r


def run_process(n,m,k,Location_matrix):
    # print(Location_matrix)
    # 随机算法
    Random_Allocation_matrix, num1 = Random_Allocation_matrix_df(n, m, k, Location_matrix)
    Allocation_matrix1 = Random_Allocation_matrix
    I_matrix1 = I_caculate(n, m, k, Allocation_matrix1)
    r1 = R_caculate(n, m, k, Allocation_matrix1, I_matrix1)

    # 贪心算法
    Allocation_matrix2, num2 = Greedy_Allocation_matrix(n, m, k, Location_matrix)
    I_matrix2 = I_caculate(n, m, k, Allocation_matrix2)
    # print(I_matrix)
    r2 = R_caculate(n, m, k, Allocation_matrix2, I_matrix2)
    # Location_matrix_show(Location_matrix)
    # Allocation_matrix_show(N, M, K,Allocation_matrix)

    # ε—贪心算法
    Allocation_matrix3, num3 = Epxilong_Greedy_Allocation_matrix(n, m, k, Location_matrix)
    I_matrix3 = I_caculate(n, m, k, Allocation_matrix3)
    # print(I_matrix)
    r3 = R_caculate(n, m, k, Allocation_matrix3, I_matrix3)
    return r1, r2, r3