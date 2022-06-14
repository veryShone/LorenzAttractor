from math import sin,cos,tan,exp,pi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
#初始條件-----------------------------------------------------------------------------------

t_a=0
t_b = 60
x0 = 10
y0 = 10
z0 = 20
σ = 10
R = 28
B = 8/3
N = 9899
'''
 x1 = r
 x2 = r_dot
 x3 = θ
 x4 = θ_dot
'''
#微分方程組---------------------------------------------------------------------------------

f = [lambda t,x1,x2,x3 : σ*(x2-x1),\
     lambda t,x1,x2,x3 : x1*(R-x3)-x2,\
     lambda t,x1,x2,x3 : x1*x2-B*x3] 
'''
 dx1/dt
 dx2/dt
 dx3/dt
'''
#ODEsolver-----------------------------------------------------------------------------------

def sysode(f,t0,tf,x0,N,method): #(積分函數,初始t,結束t,初始x,取點數,使用的積分法)
    global result
    h = (tf-t0)/N
    x0.insert(0,t0)   #把時間放到位置0
    x = x0
    t = t0
    result = [x[:]]*(N+1)         #拷貝串列的方法 佔不同記憶體位置
    #result = [[i for i in x]]*N  錯誤：佔相同記憶體位置
    #result = [x]*N               錯誤：佔相同記憶體位置
    
    '''
    使函式適用於任意數量的聯立方程組

    1.產生 "x1,x2,...,xn"、"x1+k1/2,x2+k1/2,...,xn+k1/2"、...
    2.放入"f()"中
    3.用eval將字串表達式化
    '''
    #積分法替換區：RK4----------------------------------------------------------------------
    if which_method == '1' :
        variables1 = ''
        variables2 = ''
        variables3 = ''
        variables4 = ''
        for index in range(1,len(x)):
            variables1 += 'result[i-1]['+str(index)+'],'
            variables2 += 'result[i-1]['+str(index)+']+ k1/2,'
            variables3 += 'result[i-1]['+str(index)+']+ k1/4+k2/4,'
            variables4 += 'result[i-1]['+str(index)+']+k2 + 2*k3,'
        variables1 = variables1[0:-1]
        variables2 = variables2[0:-1]
        variables3 = variables3[0:-1]
        variables4 = variables4[0:-1]
        for i in range(1,N+1):
            for xth in range(1,len(x0)):
                k1 = h*eval('f[xth-1](t    ,'+variables1+')')
                k2 = h*eval('f[xth-1](t+h/2,'+variables2+')')
                k3 = h*eval('f[xth-1](t+h/2,'+variables3+')')
                k4 = h*eval('f[xth-1](t+h  ,'+variables4+')')
                x[xth] += (1/6)*(k1+4*k3+k4)
            result[i]=x[:]
           #result[i][k]=x[k]     錯誤：佔相同記憶體位置
            t += h
            result[i][0] = t
            
    #積分法替換區：梯形法-------------------------------------------------------------------
    elif which_method == '2' :
        print('目前使用梯形法')
        y1 = ''
        y2 = ''
        for index in range(len(x)):
            y1 += 'result[i-1]['+str(index)+'],'
        y1 = y1[0:-1]
        for index in range(1,len(x)):
            y2 += 'result[i-1]['+str(index)+']+h*f['+str(index-1)+']('+y1+'),'
        y2 = y2[0:-1]

        for i in range(1,N+1):
            for k in range(1,len(x0)):
                x[k] += (h/2)*(eval('f['+str(k-1)+']('+y1+')+\
                                    f['+str(k-1)+'](t+h,'+y2+')'))
            result[i]=x[:]
           #result[i][k]=x[k]     錯誤：佔相同記憶體位置
            t += h
            result[i][0] = t
    return(result)

#使用者介面---------------------------------------------------------------------------------

which_method = input('使用\n1.RK4\n2.梯形法\n輸入選號：')
while 1:
    if which_method == '1' or which_method == '2' :break
    else:
        print('錯誤！再試一次')
        which_method = input('輸入選號：')
#呼叫sysode(f, t_a =0, t_b, [初始條件], N, 積分法)-------------------------------------------

sysode(f,t_a,t_b,\
       [x0,\
        y0,\
        z0],\
       N,
       which_method)

#作圖資料點----------------------------------------------------------------------------------
'''
x = r*cos(x3)
y = r*sin(x3)
z = r*cotα
'''
t = [result[i][0] for i in range(0,len(result))]
x = [result[i][1] for i in range(0,len(result))]
y = [result[i][2] for i in range(0,len(result))]
z = [result[i][3] for i in range(0,len(result))]
print(len(x))
#作圖----------------------------------------------------------------------------------------


fig2 = plt.figure()#3D圖

ax1 = plt.axes(projection='3d')
'''
ax1.set_xlabel('x',color="k",size=16)
ax1.set_ylabel('y',color="k",size=16)
ax1.set_zlabel('z',color="k",size=16)
ax1.plot3D(x,y,z,'b-')
'''
ln, = ax1.plot3D([], [], [], 'b-')
xdata, ydata, zdata = [], [], []
def init_1():
    ax1.set_xlim(min(x), max(x))
    ax1.set_ylim(min(y), max(y))
    ax1.set_zlim(min(z), max(z))
    return ln,
def update_1(frame):
    global xdata, ydata, zdata
    if frame <int(len(x)/10)-1:
        for i in range(10):
            xdata.append(x[10*frame+i])
            ydata.append(y[10*frame+i])
            zdata.append(z[10*frame+i])
    else: xdata, ydata, zdata = [x[0]], [y[0]], [z[0]]
    ln.set_data(xdata, ydata)
    ln.set_3d_properties(zdata)
    return ln,

ani = animation.FuncAnimation(
    fig2,  
    update_1, 
    frames= np.arange(int(len(x)/10)),
    init_func=init_1,
    interval=1,
    blit=True,
    repeat=True)
'''
#儲存影片
Writer = animation.writers['ffmpeg']
writer = Writer(fps=100, metadata=dict(artist='Me'), bitrate=5000)
ani.save("LA.mp4", writer=writer)
'''
plt.show()
