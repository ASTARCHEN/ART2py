import numpy as np
import warnings
import math
import matplotlib.pyplot as plt


def Schema_ART2_F1(I, F2_feedback, a, b, theta, e, m):

    u = np.zeros((1,m))
    p = np.zeros((1,m))
    # x = np.zeros(m)
    # w = np.zeros(m)
    # q = np.zeros(m)
    # v = np.zeros(m)
    temp = np.ones((1,m))

    while np.sum(abs(temp-u)) >= 0.001:
        temp = u[:]
        w = I + a * u
        p = u + F2_feedback
        x = w / (e + np.linalg.norm(w, 2))
        q = p / (e + np.linalg.norm(p, 2))
        x[x<theta] = 0 # 反映 f(x(k))
        q[q<theta] = 0 # 反映 f(q(k))
        v = x + b * q
        u = v /(e + np.linalg.norm(v, 2))
    return u,p


def ART_Process(I, NumNeurons, a, b, c, d, WF1_F2, WF2_F1, rho, theta, e):

    #  输入：     I - -----  待学习向量
    #  NumNeurons - -----  F2层神经元个数
    #  a, b, c, d - -----  ART2网络参数 （详见原始论文）
    #  WF1_F2 - -----  F1到F2层连接矩阵
    #  WF2_F1 - -----  F2到F1层连接矩阵
    #  rho - -----  警戒值
    #  theta, e - -----  ART2网络参数  （详见原始论文）
    #  输出：       u - ----- 用于WF1_F2, WF2_F1的学习（详见原始论文）
    #  J - ----- 本次训练胜出的神经元编号
    #
    #  使用说明： 在一个M文件中初始化NumNeurons，a, b, c, d,
    #  WF1_F2, WF2_F1，rho, theta, e等值，将输入向量放入Iput,
    #  调用ART_Process，将返回的u, J用于WF1_F2, WF2_F1的学习，学习算法可以自行选择

    y = np.zeros(NumNeurons)
    Counter = 1
    F2_feedback = 0
    m = len(I)
    while Counter > 0:
        u, p = Schema_ART2_F1(I, F2_feedback, a, b, theta, e, m)

        y[abs(y+1)>e] = (np.dot(WF1_F2[list(abs(y+1)>e),:],np.transpose(p))).transpose()[0]
        maxV = np.max(y)
        Js = np.where(y == maxV)
        J = Js[0][0]
        if abs(maxV+1) < e:
            print('分类数目已经超过最大神经元数目')
            raise
        F2_feedback = d * np.transpose(WF2_F1[:, J])
        p = u + F2_feedback
        r = (u + c * p) / (e + np.linalg.norm(u, 2) + c * np.linalg.norm(p, 2))
        R = np.linalg.norm(r, 2)
        if R < rho - e:
            y[J]= -1
            Counter = 1
        elif Counter == 1:
            Counter += 1
        else:
            # 快速学习
            WF2_F1[:, J] = np.transpose((u / (1 - d))[0])
            WF1_F2[J, :] = (u / (1 - d))[0]; # 在外部进行学习，返回u
            Counter = 0
    return u,J


def ART2(X, a=10, b=10, c=0.3, d=0.75, theta=0.1, rho=0.997, output=None, e=1E-8):
    if a <= 0:
        print('a should > 0, but a is ', str(a))
        raise
    if b <= 0:
        print('b should > 0, but  is ', str(b))
        raise
    if d < 0 or d > 1:
        print('0 <= d <= 1, but d is ', str(d))
        raise
    if c * d / (1 - d) > 1:
        print('c*d/(1-d) <= 1, but c*d/(1-d) is ', str(c * d / (1 - d)))
    if rho <= 0 or rho > 1:
        print('0 < rho <= 1, but rho is ', str(rho))
        raise
    if e > 1E-1:
        warnings('e << 1, but e is ', str(e))
    X = np.array(X)
    m, n = X.shape
    if output is None:
        output = m
    WF1_F2 = np.dot(np.ones((output,n)),0.5/((1-d)*math.sqrt(n)))
    WF2_F1 = np.zeros((n, output))

    w = np.zeros(m)
    for j in range(m):
        Input = X[j,:]
        u,J = ART_Process(Input, output,a,b,c,d,WF1_F2,WF2_F1, rho,theta,e)
        WF2_F1[:,J] = np.transpose((u/(1-d))[0])
        WF1_F2[J,:] = (u/(1-d))[0]
        w[j] = J
    return w

if __name__ == '__main__':
    # 训练x
    x1 = np.random.random((200, 2))*3 + 5
    x2 = np.hstack((np.random.random((100, 1))+2, np.random.random((100, 1))*2-4))+10
    x3 = np.hstack((np.random.random((100, 1))*2, np.random.random((100, 1))*2-2))+10
    x = np.vstack((x1,x2,x3))
    # 训练结果y
    y = np.zeros(400)
    # y[0:200] = 0
    y[200:300] = 1
    y[300:400] = 2

    e = 1E-6
    m, n = x.shape

    # 随机顺序
    ind = np.random.permutation(m)
    nx = x[ind,:]
    ny = y[ind]
    # ART2训练
    w = ART2(x,output=4, rho=0.9987)


    plt.figure(1)
    color_list_point = ['r.','b.','g.','y.','k.']
    color_list_circle = ['ro', 'bo', 'go', 'yo', 'ko']

    for j in range(int(max(y))+1):
        plt.plot(x[abs(y-j)<e,0], x[abs(y-j)<e,1],color_list_point[j])
    plt.figure(2)
    for j in range(int(max(w))+1):
        plt.plot(x[abs(w-j)<e, 0], x[abs(w-j)<e, 1], color_list_circle[j])
    plt.show()



