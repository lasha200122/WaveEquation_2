# mnbbhvcghfh

```python


import numpy as np
import matplotlib.pyplot as plt
import skimage
from matplotlib.animation import PillowWriter
from scipy.integrate import *
import cv2
import time
import glob
import os

from scipy.interpolate import interp1d

'''
Hello this python code is written for Wave Equations
Here is some good methods that you can use or modify 
for your purposes. using this methods you can create 
gif and video visualisation for your wave equation 
problem. also you can determine moving string's wave
equation using video. I think there is really interesting
problems to discuss and modify.

There also can occur some errors. The method for saving
video will not work if you do not have ffmpeg.exe in your
computer. you can download it from this link: https://ffmpeg.org/download.html
do not forget to input the path of ffmpeg.exe in the input fields later ;)

For better understanding this algorithms let's talk about annotations.
Our Wave Equation -> u_tt = c^2 u_xx + f(x,t) 
this methods is considered for boundary, initial value problems so there
we will have: u(0,t) , u(L,t) , u(x,0), u_t(x,0) and also f(x,t)

****  WaveSolver attributes ****
I -> Function for u(x,0)
V -> Function for u_t(x,0)
f -> Function for f(x,t)
c -> Speed of wave
L -> Length of wave
dt -> Delta t
dx -> Delta x
Cfl -> Constant (computation Cfl = c * dt / dx  )
T -> Time (duration)
Amax -> maximum amplitude of given wave equation
ffmpeg -> path of ffmpeg.exe
Func -> another function for wave equation
path -> video path 


Solver method solves wave equation using finite differences.

viz method visualizes solution of wave equation.

RemovePictures method removes all pictures that where stored while using viz method (nothing important just fast way to 
delete junked files).

ExactEquation method is formula u_e(x,t) = A sin(pi x / L) cos(pi c t / L) and returns it's value at the current t and 
domain.

Analysis method returns analytical solution of wave equation. (mostly this is used to check correctness of numerical 
solutions).

WithMyFunction method return also analytical solution of wave equation. as you know there are too many ways to solve
wave equation so this method returns solutions of separation variables 

NumericalGif method saves numerical solution as gif

AnalysisGif method saves analysis solution as gif

CompareGif method saves analysis and numerical solutions as one gif

PlotError method calculates error between numerical and analysis solutions and then plots it as graph and saves it as 
png.
'''

"Let's Create input attributes and functions"


def I(x):
    return np.sin(2*np.pi*x/3)


def V(x):
    return np.sin(2*np.pi*x/3) * np.pi*2 / 3


def Func(x,t):
    return (np.cos(2*np.pi*t/3) + np.sin(2*np.pi*t/3)) * np.sin(2 * np.pi * x / 3)


f = 0
c = 1
L = 3
dt = 0.01
Cfl = 0.5
T = 3
Amax = 3
ffmpeg = 'C:/Users/ELENE/Desktop/ffmpeg-5.1-full_build/bin/ffmpeg.exe'
video_path = 'movie.ogg'

## Method for solving Wave Equation

def solver(I, V, f, c, L, dt, Cfl, T, user_action=None):
    Nt = int(round(T / dt))
    t = np.linspace(0, Nt * dt, Nt + 1)
    dx = dt * c / float(Cfl)
    Nx = int(round(L / dx))
    x = np.linspace(0, L, Nx + 1)
    C2 = Cfl ** 2
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    if f is None or f == 0:
        f = lambda x, t: 0
        if V is None or V == 0:
            V = lambda x: 0
    u = np.zeros(Nx + 1)
    u_n = np.zeros(Nx + 1)
    u_nm1 = np.zeros(Nx + 1)
    for i in range(0, Nx + 1):
        u_n[i] = I(x[i])
    if user_action is not None:
        user_action(u_n, x, t, 0)
    n = 0
    for i in range(1, Nx):
        u[i] = u_n[i] + dt * V(x[i]) + 0.5 * C2 * (u_n[i - 1] - 2 * u_n[i] + u_n[i + 1]) + 0.5 * dt ** 2 * f(x[i],t[n])
    u[0] = 0; u[Nx] = 0
    if user_action is not None:
        user_action(u, x, t, 1)
    u_nm1[:] = u_n; u_n[:] = u
    for n in range(1, Nt):
        for i in range(1, Nx):
            u[i] = - u_nm1[i] + 2 * u_n[i] + C2 * (u_n[i - 1] - 2 * u_n[i] + u_n[i + 1]) + dt ** 2 * f(x[i], t[n])
        u[0] = 0; u[Nx] = 0
        if user_action is not None:
            if user_action(u, x, t, n + 1):
                break
        u_nm1[:] = u_n; u_n[:] = u
    return u, x, t

#Method for visualizing Wave Equation

def viz(I, V, f, c, L, dt, Cfl, T, animate=True, solver_function=solver):
    def plot_u_st(u, x, t, n):
        plt.plot(x, u, 'r--', xlabel='x', ylabel='u', axis=[0, L, -Amax, Amax], title='t=%f' % t[n], show=True)
        time.sleep(2) if t[n] == 0 else time.sleep(0.2)
        plt.savefig("frame_%04d.png" % n)
    result = []
    class PlotMatplotlib:
        def __call__(self, u, x, t, n):
            result.append(list(u))
            if n == 0:
                plt.ion()
                self.lines = plt.plot(x, u)
                # you can also turn border and axis on.
                plt.box(False)
                plt.axis('off')
                plt.xlabel('x'); plt.ylabel('u')
                plt.axis([0, L, -Amax, Amax])
                # you can add this for timer
                # plt.legend(['t=%f' % t[n]], loc='lower left')
            else:
                self.lines[0].set_ydata(u)
                # plt.legend(['t=%f' % t[n]], loc='lower left')
                plt.draw()
            time.sleep(2) if t[n] == 0 else time.sleep(0.2)
            plt.savefig('tmp_%04d.png' % n)
    plot_u = PlotMatplotlib()
    for filename in glob.glob("tmp_*.png"):
        os.remove(filename)
    user_action = plot_u if animate else None
    u, x, t = solver_function(I, V, f, c, L, dt, Cfl, T, user_action)
    fps = int(1/dt)
    codec2ext = dict(libtheora='ogg')
    filespec = 'tmp_%04d.png'
    for codec in codec2ext:
        ext = codec2ext[codec]
        cmd = ffmpeg+' -r %(fps)d -i %(filespec)s -vcodec %(codec)s movie.%(ext)s' % vars()
        os.system(cmd)
    return result

class WaveSolver:
    def __init__(self, I, V, f, c, L, dt, Cfl, T, Amax, ffmpeg, Func = None ,user_action=None):
        self.I = I
        self.V = V
        self.f = f
        self.c = c
        self.L = L
        self.dt = dt
        self.Cfl = Cfl
        self.T = T
        self.fps = int(1/self.dt)
        self.Amax = Amax
        self.ffmpeg = ffmpeg
        self.Func = Func
        self.user_action = user_action
        self.Nt = int(round(self.T / self.dt))
        self.time = np.linspace(0, self.Nt * self.dt, self.Nt + 1)
        self.dx = self.dt * self.c / float(self.Cfl)
        self.Nx = int(round(self.L / self.dx))
        self.domain = np.linspace(0, self.L, self.Nx + 1)

        ## Solving Part
        self.numericalGrid = viz(self.I, self.V, self.f, self.c, self.L, self.dt, self.Cfl, self.T)
        if self.Func == None:
            self.analysisGrid = self.Analysis()
        else:
            self.analysisGrid = self.WithMyFunction()

        ### Saving Part
        # self.AnalysisGif()
        # self.NumericalGif()
        # self.CompareGif()
        # self.PlotError()

        ### Printing Part
        print()
        print("Speed: " + str(self.c))
        print("Length: " + str(self.L))
        print("Delta x: " + str(self.dx))
        print("Delta t: " + str(self.dt))
        print("Time: " + str(self.T))
        print("Constant: " + str(self.Cfl))
        print("Amplitude: " + str(self.Amax))
        print("Time intervals: " + str(self.Nt))
        print("Domain intervals: " + str(self.Nx))
        print("FPS: " + str(self.fps))
        print()

    def RemovePictures(self):
        for filename in glob.glob("tmp_*.png"):
            os.remove(filename)
        return True

    def ExactEquation(self, A, x, c, L, t):
        return A * np.sin(np.pi * x / L) * np.cos(np.pi * c * t / L)

    def Analysis(self):
        grid = []
        for t in self.time:
            ls = self.ExactEquation(self.Amax, self.domain, self.c, self.L, t)
            grid.append(ls)
        return grid

    def WithMyFunction(self):
        grid = []
        for t in self.time:
            ls = self.Func(self.domain, t)
            grid.append(ls)
        return grid

    def NumericalGif(self):
        print()
        print("Saving Numerical Gif")
        metadata = dict(title="NumericalSolution", artist="Lasha_Shavgulidze")
        fps = int(1/self.dt)
        writer = PillowWriter(fps=fps, metadata=metadata)
        fig = plt.figure()
        graph, = plt.plot([], [], label="Numerical")
        plt.legend()
        plt.ylim(-self.Amax, self.Amax)
        plt.xlim(0, self.L)
        with writer.saving(fig, "Numerical.gif", 100):
            for i in self.numericalGrid:
                graph.set_data(self.domain,i)
                writer.grab_frame()
        print()
        print("Gif Saved Successfully")
        return

    def AnalysisGif(self):
        print()
        print("Saving Analysis Gif")
        metadata = dict(title="AnalysisSolution", artist="Lasha_Shavgulidze")
        fps = int(1/self.dt)
        writer = PillowWriter(fps=fps, metadata=metadata)
        fig = plt.figure()
        graph, = plt.plot([], [], label="Analysis")
        plt.legend()
        plt.ylim(-self.Amax, self.Amax)
        plt.xlim(0, self.L)
        with writer.saving(fig, "Analysis.gif", 100):
            for i in self.analysisGrid:
                graph.set_data(self.domain,i)
                writer.grab_frame()
        print()
        print("Gif Saved Successfully")
        return

    def CompareGif(self):
        print()
        print("Saving Compare Gif")
        metadata = dict(title="Compare", artist="Lasha_Shavgulidze")
        fps = int(1 / self.dt)
        writer = PillowWriter(fps=fps, metadata=metadata)
        fig = plt.figure()
        graph, = plt.plot([], [], label="Numerical")
        graph2, = plt.plot([], [], label="Analysis")
        plt.legend()
        plt.ylim(-self.Amax, self.Amax)
        plt.xlim(0, self.L)
        with writer.saving(fig, "Compare.gif", 100):
            for i in range(len(self.numericalGrid)):
                graph.set_data(self.domain, self.numericalGrid[i])
                graph2.set_data(self.domain, self.analysisGrid[i])
                writer.grab_frame()
        print()
        print("Gif Saved Successfully")
        return

    def PlotError(self):
        y = []
        x = []
        for j in range(len(self.analysisGrid)):
            diff = self.analysisGrid[j] - self.numericalGrid[j]
            val = np.sqrt(sum([i ** 2 for i in diff]) * self.dx)
            y.append(val)
            x.append(j)
        plt.figure()
        plt.plot(x, y, color='red')
        plt.savefig("Error.png")
        return



class WaveFinder:
    def __init__(self, path, L,A,n):
        self.path = path
        self.n = n
        self.L = L
        self.A = A
        self.frames = {}
        self.fps = 0
        self.frame_count = 0
        self.dt = 0
        self.dx = 0
        self.time = 0
        self.shape = (0, 0)
        self.start = (0, 0)
        self.end = (0, 0)
        self.domain = np.arange(0,self.L + self.L/n, self.L/n)
        self.difference = 0
        self.constants = []
        self.coordinates = []
        self.FindValues()
        self.FindPoints()
        self.scale = self.L / abs(self.start[1] - self.end[1])
        self.GetGrid()
        self.Constant()

        self.C2 = np.sqrt(abs(sum(self.constants) / len(self.constants)))
        # self.C2 = np.sqrt(sum([i ** 2 for i in self.constants]) * self.dx)
        self.speed = self.C2 * self.dx / self.dt


        ###Printing Part
        print("Video path: " + self.path)
        print("Video duration: " + str(self.time))
        print("Frames: " + str(self.frame_count))
        print("Fps: " + str(self.fps))
        print("Video shape: " + str(self.shape))
        print("Wave length: " + str(self.L))
        print("Delta t: " + str(self.dt))
        print("Delta x: " + str(self.dx))
        print("Starting Coordinate: " + str(self.start))
        print("Ending Coordinate: " + str(self.end))
        print("Difference between x-coordinates: " + str(self.difference))
        print("Number of x-coordinates: " + str(len(self.coordinates[0])))
        print("Scale: " + str(self.scale))
        print("Constant: " + str(self.C2))
        print("Speed: " + str(self.speed))

        ###

    def FindValues(self):
        # Create a VideoCapture object and read from input file
        cap = cv2.VideoCapture(self.path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.time = self.frame_count / self.fps
        self.dt = self.time / self.frame_count
        # Check if camera opened successfully
        if (cap.isOpened() == False):
            print("Error opening video stream or file")
        print("Getting Data")
        print("Frame count: " + str(self.frame_count))
        # Read until video is completed
        count = 0
        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # edges = cv2.medianBlur(frame,3)
                # edges = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 5)
                # edges = cv2.Laplacian(edges, -1, ksize=5, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
                blur = cv2.GaussianBlur(frame,(0,0), sigmaX=33, sigmaY=33)
                edges = cv2.divide(frame, blur, scale=255)
                edges = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                edges = 255 - edges
                if count < 1:
                    print("Shape of frame: " + str(edges.shape))
                    print("Reading frames...")
                    self.shape = edges.shape
                count += 1
                edges[edges != 255] = 0
                edges[edges >= 255] = 255
                self.frames[count] = edges
                # cv2.imshow("video",edges)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            # Break the loop
            else:
                break
        print()
        print("Finished reading frames")
        return

    def FindPoints(self):
        print()
        print("Finding starting and ending points coordinates")
        a = None
        b = None
        index = 0
        while a == None or b == None:
            if a == None:
                for i in range(len(self.frames[1])):
                    if self.frames[1][i][index] == 255:
                        a = (i,index)
            if b == None:
                for i in range(len(self.frames[1])):
                    if self.frames[1][i][len(self.frames[1][i])-1-index] == 255:
                        b = (i,len(self.frames[1][i])-1-index)
            index += 1
        self.start = a
        self.end = b
        self.difference = abs(a[1] - b[1])
        self.scale = self.L / self.difference
        return

    def GetGrid(self):
        print()
        print("Creating Grid...")
        for key in self.frames.keys():
            coords = np.column_stack(np.where(self.frames[key] >= 255))
            X = []
            Y = []
            x = [self.scale * (i[1] - self.start[1]) for i in coords]
            y = [(self.start[0] - i[0]) * self.scale * self.A for i in coords]

            while len(x) != 0:
                indexes = []
                value = x[0]
                sum = 0
                X.append(value)
                for i in range(len(x)):
                    if x[i] == value:
                        indexes.append(i)
                for index in sorted(indexes, reverse=True):
                    sum += y[index]
                    del y[index]
                    del x[index]
                sum = sum / len(indexes)
                Y.append(sum)
            Y = [x for _, x in sorted(zip(X, Y))]
            X = sorted(X)
            f_is = interp1d(X, Y, kind="quadratic")
            yy = f_is(self.domain)
            self.coordinates.append(yy)

        # step = abs(self.start[1] - self.end[1]) // self.n
        #allData = []
        # for key in [1,2]:
        #     data =[]
        #     for i in range(len(self.frames[key])):
        #         for j in range(len(self.frames[key][i])):
        #             # if len(data) < self.n:
        #             if self.frames[key][i][j] == 255:
        #                 if not j in [z[1] for z in data]:
        #                     data.append((i,j))
        #     # allData.append(data)
        #     x = np.array([v[1] - self.start[1] - 1 for v in data])
        #     y = np.array([self.start[0] - v[0] for v in data])
        #     f_is = interp1d(x* self.scale,y* self.scale, kind='cubic')
        #     yy = f_is(self.domain)
        #     print(yy)
        #     self.solution.append(yy)
            # plt.plot(self.domain,yy)
            # plt.show()


            # Y = []
            # frame = self.frames[key][:,self.start[1]:self.end[1]]
            # x = 0
            # while len(Y) < self.n:
            #     count = 0
            #     for row in range(len(frame)):
            #         if count == 0:
            #             if frame[row][x] == 255:
            #                 val = self.start[0] - row
            #                 Y.append(val)
            #                 count += 1
            #     x += step
            self.dx = self.domain[1] - self.domain[0]
        return

    def Constant(self):
        print()
        print("Finding Constants")
        for n in range(1, len(self.coordinates) - 1):
            for i in range(1, len(self.coordinates[n]) - 1):
                part2 = self.coordinates[n][i + 1] - 2 * self.coordinates[n][i] + self.coordinates[n][i - 1]
                if part2 != 0:
                    C2 = (self.coordinates[n + 1][i] + self.coordinates[n - 1][i] - 2 * self.coordinates[n][i]) / part2
                    self.constants.append(C2)

                else:
                    C2 = abs(self.coordinates[n + 1][i] + self.coordinates[n - 1][i] - 2 * self.coordinates[n][i])
        return
        # print()
        # print("Finding Constants")
        # print(self.coordinates)
        # for i in range(1, len(self.coordinates[0]) -1):
        #     C = 2 * (self.coordinates[0][i] - self.coordinates[1][i]) / (self.coordinates[0][i+1] - 2* self.coordinates[0][i] + self.coordinates[0][i-1])
        #     if C < 1 and C > -1:
        #         print(C)
        #         self.constants.append(C)
        # for n in range(1, len(self.solution) - 1):
        #     for i in range(1, len(self.solution[n]) - 1):
        #         part2 = self.solution[n][i + 1] - 2 * self.solution[n][i] + self.solution[n][i - 1]
        #         if part2 != 0:
        #             C2 = (self.solution[n + 1][i] + self.solution[n - 1][i] - 2 * self.solution[n][i]) / part2
        #             self.constants.append(C2)
        #
        #         # else:
        #         #     C2 = abs(self.solution[n + 1][i] + self.solution[n - 1][i] - 2 * self.solution[n][i])
        # return

#
# Real = WaveSolver(I, V, f, c, L, dt, Cfl, T, Amax, ffmpeg, Func)
# Real.RemovePictures()
# sol1 = Real.numericalGrid
# dom1 = Real.domain
WaveFinder(video_path,L, 2,150)
# Test = WaveSolver(I,V,f,1.5430020287394293, L, 0.01, 0.2571670047899049,T,Amax,ffmpeg,Func)
# dom2 = Test.domain
# sol2 = Test.numericalGrid
# Test.RemovePictures()
# print()
# print("Saving Interesting Gif")
# metadata = dict(title="Please", artist="Lasha_Shavgulidze")
# fps = int(1 / 0.01)
# writer = PillowWriter(fps=fps, metadata=metadata)
# fig = plt.figure()
# graph, = plt.plot([], [], label="Numerical")
# graph2, = plt.plot([], [], label="Analysis")
# plt.legend()
# plt.ylim(-Amax, Amax)
# plt.xlim(0, L)
# with writer.saving(fig, "Please.gif", 100):
#     for i in range(len(sol1)):
#         graph.set_data(dom1, sol1[i])
#         graph2.set_data(dom2, sol2[i])
#         writer.grab_frame()
# print()
# print("Gif Saved Successfully")
#


'''
* References

Book -> Finite Difference Computing with PDEs (Hans Petter Lagtangen - Svein Linge)
'''



```

### Second 

```python

import numpy as np
import matplotlib.pyplot as plt
import skimage
from matplotlib.animation import PillowWriter
from scipy.integrate import *
import cv2
import time
import glob
import os

from scipy.interpolate import interp1d

'''
Hello this python code is written for Wave Equations
Here is some good methods that you can use or modify 
for your purposes. using this methods you can create 
gif and video visualisation for your wave equation 
problem. also you can determine moving string's wave
equation using video. I think there is really interesting
problems to discuss and modify.

There also can occur some errors. The method for saving
video will not work if you do not have ffmpeg.exe in your
computer. you can download it from this link: https://ffmpeg.org/download.html
do not forget to input the path of ffmpeg.exe in the input fields later ;)

For better understanding this algorithms let's talk about annotations.
Our Wave Equation -> u_tt = c^2 u_xx + f(x,t) 
this methods is considered for boundary, initial value problems so there
we will have: u(0,t) , u(L,t) , u(x,0), u_t(x,0) and also f(x,t)

****  WaveSolver attributes ****
I -> Function for u(x,0)
V -> Function for u_t(x,0)
f -> Function for f(x,t)
c -> Speed of wave
L -> Length of wave
dt -> Delta t
dx -> Delta x
Cfl -> Constant (computation Cfl = c * dt / dx  )
T -> Time (duration)
Amax -> maximum amplitude of given wave equation
ffmpeg -> path of ffmpeg.exe
Func -> another function for wave equation
path -> video path 


Solver method solves wave equation using finite differences.

viz method visualizes solution of wave equation.

RemovePictures method removes all pictures that where stored while using viz method (nothing important just fast way to 
delete junked files).

ExactEquation method is formula u_e(x,t) = A sin(pi x / L) cos(pi c t / L) and returns it's value at the current t and 
domain.

Analysis method returns analytical solution of wave equation. (mostly this is used to check correctness of numerical 
solutions).

WithMyFunction method return also analytical solution of wave equation. as you know there are too many ways to solve
wave equation so this method returns solutions of separation variables 

NumericalGif method saves numerical solution as gif

AnalysisGif method saves analysis solution as gif

CompareGif method saves analysis and numerical solutions as one gif

PlotError method calculates error between numerical and analysis solutions and then plots it as graph and saves it as 
png.
'''

"Let's Create input attributes and functions"


def I(x):
    return np.sin(2*np.pi*x/3)


def V(x):
    return np.sin(2*np.pi*x/3) * np.pi*2 / 3


def Func(x,t):
    return (np.cos(2*np.pi*t/3) + np.sin(2*np.pi*t/3)) * np.sin(2 * np.pi * x / 3)


f = 0
c = 1
L = 3
dt = 0.01
Cfl = 0.5
T = 3
Amax = 3
ffmpeg = 'C:/Users/ELENE/Desktop/ffmpeg-5.1-full_build/bin/ffmpeg.exe'
video_path = 'movie.ogg'

## Method for solving Wave Equation

def solver(I, V, f, c, L, dt, Cfl, T, user_action=None):
    Nt = int(round(T / dt))
    t = np.linspace(0, Nt * dt, Nt + 1)
    dx = dt * c / float(Cfl)
    Nx = int(round(L / dx))
    x = np.linspace(0, L, Nx + 1)
    C2 = Cfl ** 2
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    if f is None or f == 0:
        f = lambda x, t: 0
        if V is None or V == 0:
            V = lambda x: 0
    u = np.zeros(Nx + 1)
    u_n = np.zeros(Nx + 1)
    u_nm1 = np.zeros(Nx + 1)
    for i in range(0, Nx + 1):
        u_n[i] = I(x[i])
    if user_action is not None:
        user_action(u_n, x, t, 0)
    n = 0
    for i in range(1, Nx):
        u[i] = u_n[i] + dt * V(x[i]) + 0.5 * C2 * (u_n[i - 1] - 2 * u_n[i] + u_n[i + 1]) + 0.5 * dt ** 2 * f(x[i],t[n])
    u[0] = 0; u[Nx] = 0
    if user_action is not None:
        user_action(u, x, t, 1)
    u_nm1[:] = u_n; u_n[:] = u
    for n in range(1, Nt):
        for i in range(1, Nx):
            u[i] = - u_nm1[i] + 2 * u_n[i] + C2 * (u_n[i - 1] - 2 * u_n[i] + u_n[i + 1]) + dt ** 2 * f(x[i], t[n])
        u[0] = 0; u[Nx] = 0
        if user_action is not None:
            if user_action(u, x, t, n + 1):
                break
        u_nm1[:] = u_n; u_n[:] = u
    return u, x, t

#Method for visualizing Wave Equation

def viz(I, V, f, c, L, dt, Cfl, T, animate=True, solver_function=solver):
    def plot_u_st(u, x, t, n):
        plt.plot(x, u, 'r--', xlabel='x', ylabel='u', axis=[0, L, -Amax, Amax], title='t=%f' % t[n], show=True)
        time.sleep(2) if t[n] == 0 else time.sleep(0.2)
        plt.savefig("frame_%04d.png" % n)
    result = []
    class PlotMatplotlib:
        def __call__(self, u, x, t, n):
            result.append(list(u))
            if n == 0:
                plt.ion()
                self.lines = plt.plot(x, u)
                # you can also turn border and axis on.
                plt.box(False)
                plt.axis('off')
                plt.xlabel('x'); plt.ylabel('u')
                plt.axis([0, L, -Amax, Amax])
                # you can add this for timer
                # plt.legend(['t=%f' % t[n]], loc='lower left')
            else:
                self.lines[0].set_ydata(u)
                # plt.legend(['t=%f' % t[n]], loc='lower left')
                plt.draw()
            time.sleep(2) if t[n] == 0 else time.sleep(0.2)
            plt.savefig('tmp_%04d.png' % n)
    plot_u = PlotMatplotlib()
    for filename in glob.glob("tmp_*.png"):
        os.remove(filename)
    user_action = plot_u if animate else None
    u, x, t = solver_function(I, V, f, c, L, dt, Cfl, T, user_action)
    fps = int(1/dt)
    codec2ext = dict(libtheora='ogg')
    filespec = 'tmp_%04d.png'
    for codec in codec2ext:
        ext = codec2ext[codec]
        cmd = ffmpeg+' -r %(fps)d -i %(filespec)s -vcodec %(codec)s movie.%(ext)s' % vars()
        os.system(cmd)
    return result

class WaveSolver:
    def __init__(self, I, V, f, c, L, dt, Cfl, T, Amax, ffmpeg, Func = None ,user_action=None):
        self.I = I
        self.V = V
        self.f = f
        self.c = c
        self.L = L
        self.dt = dt
        self.Cfl = Cfl
        self.T = T
        self.fps = int(1/self.dt)
        self.Amax = Amax
        self.ffmpeg = ffmpeg
        self.Func = Func
        self.user_action = user_action
        self.Nt = int(round(self.T / self.dt))
        self.time = np.linspace(0, self.Nt * self.dt, self.Nt + 1)
        self.dx = self.dt * self.c / float(self.Cfl)
        self.Nx = int(round(self.L / self.dx))
        self.domain = np.linspace(0, self.L, self.Nx + 1)

        ## Solving Part
        self.numericalGrid = viz(self.I, self.V, self.f, self.c, self.L, self.dt, self.Cfl, self.T)
        if self.Func == None:
            self.analysisGrid = self.Analysis()
        else:
            self.analysisGrid = self.WithMyFunction()

        ### Saving Part
        # self.AnalysisGif()
        # self.NumericalGif()
        # self.CompareGif()
        # self.PlotError()

        ### Printing Part
        print()
        print("Speed: " + str(self.c))
        print("Length: " + str(self.L))
        print("Delta x: " + str(self.dx))
        print("Delta t: " + str(self.dt))
        print("Time: " + str(self.T))
        print("Constant: " + str(self.Cfl))
        print("Amplitude: " + str(self.Amax))
        print("Time intervals: " + str(self.Nt))
        print("Domain intervals: " + str(self.Nx))
        print("FPS: " + str(self.fps))
        print()

    def RemovePictures(self):
        for filename in glob.glob("tmp_*.png"):
            os.remove(filename)
        return True

    def ExactEquation(self, A, x, c, L, t):
        return A * np.sin(np.pi * x / L) * np.cos(np.pi * c * t / L)

    def Analysis(self):
        grid = []
        for t in self.time:
            ls = self.ExactEquation(self.Amax, self.domain, self.c, self.L, t)
            grid.append(ls)
        return grid

    def WithMyFunction(self):
        grid = []
        for t in self.time:
            ls = self.Func(self.domain, t)
            grid.append(ls)
        return grid

    def NumericalGif(self):
        print()
        print("Saving Numerical Gif")
        metadata = dict(title="NumericalSolution", artist="Lasha_Shavgulidze")
        fps = int(1/self.dt)
        writer = PillowWriter(fps=fps, metadata=metadata)
        fig = plt.figure()
        graph, = plt.plot([], [], label="Numerical")
        plt.legend()
        plt.ylim(-self.Amax, self.Amax)
        plt.xlim(0, self.L)
        with writer.saving(fig, "Numerical.gif", 100):
            for i in self.numericalGrid:
                graph.set_data(self.domain,i)
                writer.grab_frame()
        print()
        print("Gif Saved Successfully")
        return

    def AnalysisGif(self):
        print()
        print("Saving Analysis Gif")
        metadata = dict(title="AnalysisSolution", artist="Lasha_Shavgulidze")
        fps = int(1/self.dt)
        writer = PillowWriter(fps=fps, metadata=metadata)
        fig = plt.figure()
        graph, = plt.plot([], [], label="Analysis")
        plt.legend()
        plt.ylim(-self.Amax, self.Amax)
        plt.xlim(0, self.L)
        with writer.saving(fig, "Analysis.gif", 100):
            for i in self.analysisGrid:
                graph.set_data(self.domain,i)
                writer.grab_frame()
        print()
        print("Gif Saved Successfully")
        return

    def CompareGif(self):
        print()
        print("Saving Compare Gif")
        metadata = dict(title="Compare", artist="Lasha_Shavgulidze")
        fps = int(1 / self.dt)
        writer = PillowWriter(fps=fps, metadata=metadata)
        fig = plt.figure()
        graph, = plt.plot([], [], label="Numerical")
        graph2, = plt.plot([], [], label="Analysis")
        plt.legend()
        plt.ylim(-self.Amax, self.Amax)
        plt.xlim(0, self.L)
        with writer.saving(fig, "Compare.gif", 100):
            for i in range(len(self.numericalGrid)):
                graph.set_data(self.domain, self.numericalGrid[i])
                graph2.set_data(self.domain, self.analysisGrid[i])
                writer.grab_frame()
        print()
        print("Gif Saved Successfully")
        return

    def PlotError(self):
        y = []
        x = []
        for j in range(len(self.analysisGrid)):
            diff = self.analysisGrid[j] - self.numericalGrid[j]
            val = np.sqrt(sum([i ** 2 for i in diff]) * self.dx)
            y.append(val)
            x.append(j)
        plt.figure()
        plt.plot(x, y, color='red')
        plt.savefig("Error.png")
        return



class WaveFinder:
    def __init__(self, path, L,A,n):
        self.path = path
        self.n = n
        self.L = L
        self.A = A
        self.frames = {}
        self.fps = 0
        self.frame_count = 0
        self.dt = 0
        self.dx = 0
        self.time = 0
        self.shape = (0, 0)
        self.start = (0, 0)
        self.end = (0, 0)
        self.domain = np.arange(0,self.L + self.L/n, self.L/n)
        self.difference = 0
        self.constants = []
        self.coordinates = []
        self.FindValues()
        self.FindPoints()
        self.scale = self.L / abs(self.start[1] - self.end[1])
        self.GetGrid()
        self.Constant()
        print(self.constants)
        self.C2 = np.sqrt(abs(sum(self.constants) / len(self.constants)))
        # self.C2 = np.sqrt(sum([i ** 2 for i in self.constants]) * self.dx)
        self.speed = self.C2 * self.dx / self.dt


        ###Printing Part
        print("Video path: " + self.path)
        print("Video duration: " + str(self.time))
        print("Frames: " + str(self.frame_count))
        print("Fps: " + str(self.fps))
        print("Video shape: " + str(self.shape))
        print("Wave length: " + str(self.L))
        print("Delta t: " + str(self.dt))
        print("Delta x: " + str(self.dx))
        print("Starting Coordinate: " + str(self.start))
        print("Ending Coordinate: " + str(self.end))
        print("Difference between x-coordinates: " + str(self.difference))
        print("Number of x-coordinates: " + str(len(self.coordinates[0])))
        print("Scale: " + str(self.scale))
        print("Constant: " + str(self.C2))
        print("Speed: " + str(self.speed))

        ###

    def FindValues(self):
        # Create a VideoCapture object and read from input file
        cap = cv2.VideoCapture(self.path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.time = self.frame_count / self.fps
        self.dt = self.time / self.frame_count
        # Check if camera opened successfully
        if (cap.isOpened() == False):
            print("Error opening video stream or file")
        print("Getting Data")
        print("Frame count: " + str(self.frame_count))
        # Read until video is completed
        count = 0
        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True and count < 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # edges = cv2.medianBlur(frame,3)
                # edges = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 5)
                # edges = cv2.Laplacian(edges, -1, ksize=5, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
                blur = cv2.GaussianBlur(frame,(0,0), sigmaX=33, sigmaY=33)
                edges = cv2.divide(frame, blur, scale=255)
                edges = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                edges = 255 - edges
                if count < 1:
                    print("Shape of frame: " + str(edges.shape))
                    print("Reading frames...")
                    self.shape = edges.shape
                count += 1
                edges[edges != 255] = 0
                edges[edges >= 255] = 255
                self.frames[count] = edges
                # cv2.imshow("video",edges)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            # Break the loop
            else:
                break
        print()
        print("Finished reading frames")
        return

    def FindPoints(self):
        print()
        print("Finding starting and ending points coordinates")
        a = None
        b = None
        index = 0
        while a == None or b == None:
            if a == None:
                for i in range(len(self.frames[1])):
                    if self.frames[1][i][index] == 255:
                        a = (i,index)
            if b == None:
                for i in range(len(self.frames[1])):
                    if self.frames[1][i][len(self.frames[1][i])-1-index] == 255:
                        b = (i,len(self.frames[1][i])-1-index)
            index += 1
        self.start = a
        self.end = b
        self.difference = abs(a[1] - b[1])
        self.scale = self.L / self.difference
        return

    def GetGrid(self):
        print()
        print("Creating Grid...")
        for key in self.frames.keys():
            coords = np.column_stack(np.where(self.frames[key] >= 255))
            X = []
            Y = []
            x = [(i[1] - self.start[1]) * self.scale for i in coords if i[1] >= self.start[1]] #
            y = [(self.start[0] - i[0])* self.scale for i in coords if i[1]>= self.start[1]] #* self.scale * self.A

            while len(x) != 0:
                indexes = []
                value = x[0]
                sum = 0
                X.append(value)
                for i in range(len(x)):
                    if x[i] == value:
                        indexes.append(i)
                for index in sorted(indexes, reverse=True):
                    sum += y[index]
                    del y[index]
                    del x[index]
                sum = sum / len(indexes)
                Y.append(sum)
            Y = [x for _, x in sorted(zip(X, Y))]
            X = sorted(X)
            f_is = interp1d(X, Y, kind="linear")
            yy = f_is(self.domain)
            print(yy)
            self.coordinates.append(list(yy))

        # step = abs(self.start[1] - self.end[1]) // self.n
        #allData = []
        # for key in [1,2]:
        #     data =[]
        #     for i in range(len(self.frames[key])):
        #         for j in range(len(self.frames[key][i])):
        #             # if len(data) < self.n:
        #             if self.frames[key][i][j] == 255:
        #                 if not j in [z[1] for z in data]:
        #                     data.append((i,j))
        #     # allData.append(data)
        #     x = np.array([v[1] - self.start[1] - 1 for v in data])
        #     y = np.array([self.start[0] - v[0] for v in data])
        #     f_is = interp1d(x* self.scale,y* self.scale, kind='cubic')
        #     yy = f_is(self.domain)
        #     print(yy)
        #     self.solution.append(yy)
            # plt.plot(self.domain,yy)
            # plt.show()


            # Y = []
            # frame = self.frames[key][:,self.start[1]:self.end[1]]
            # x = 0
            # while len(Y) < self.n:
            #     count = 0
            #     for row in range(len(frame)):
            #         if count == 0:
            #             if frame[row][x] == 255:
            #                 val = self.start[0] - row
            #                 Y.append(val)
            #                 count += 1
            #     x += step
            self.dx = self.domain[1] - self.domain[0]
        return

    def Constant(self):
        # print()
        # print("Finding Constants")
        # for n in range(1, len(self.coordinates) - 1):
        #     for i in range(1, len(self.coordinates[n]) - 1):
        #         part2 = self.coordinates[n][i + 1] - 2 * self.coordinates[n][i] + self.coordinates[n][i - 1]
        #         if part2 != 0:
        #             C2 = (self.coordinates[n + 1][i] + self.coordinates[n - 1][i] - 2 * self.coordinates[n][i]) / part2
        #             self.constants.append(C2)
        #
        #         else:
        #             C2 = abs(self.coordinates[n + 1][i] + self.coordinates[n - 1][i] - 2 * self.coordinates[n][i])
        # return
        print()
        print("Finding Constants")
        print(self.coordinates)
        self.coordinates[0][0] = 0
        self.coordinates[1][0] = 0
        self.coordinates[0][len(self.coordinates[0])-1] = 0
        self.coordinates[0][len(self.coordinates[0])-1] = 0
        for i in range(1, len(self.coordinates[0]) -1):
            p = self.coordinates[0][i+1] - 2* self.coordinates[0][i] + self.coordinates[0][i-1]
            if p !=0:
                C = 2 * (self.coordinates[0][i] - self.coordinates[1][i]) / p
                print(C)
                if C < 1 and C > -1 and C!=0:
                    self.constants.append((C))



# #
# Real = WaveSolver(I, V, f, c, L, dt, Cfl, T, Amax, ffmpeg, Func)
# Real.RemovePictures()
# sol1 = Real.numericalGrid
# dom1 = Real.domain
WaveFinder(video_path,L, 2,150)
# Test = WaveSolver(I,V,f,0.9427325117681262, L, 0.01, 0.471366255884063,T,Amax,ffmpeg,Func)
# dom2 = Test.domain
# sol2 = Test.numericalGrid
# Test.RemovePictures()
# print()
# print("Saving Interesting Gif")
# metadata = dict(title="Please", artist="Lasha_Shavgulidze")
# fps = int(1 / 0.01)
# writer = PillowWriter(fps=fps, metadata=metadata)
# fig = plt.figure()
# graph, = plt.plot([], [], label="Numerical")
# graph2, = plt.plot([], [], label="Analysis")
# plt.legend()
# plt.ylim(-Amax, Amax)
# plt.xlim(0, L)
# with writer.saving(fig, "Please.gif", 100):
#     for i in range(len(sol1)):
#         graph.set_data(dom1, sol1[i])
#         graph2.set_data(dom2, sol2[i])
#         writer.grab_frame()
# print()
# print("Gif Saved Successfully")
#


'''
* References

Book -> Finite Difference Computing with PDEs (Hans Petter Lagtangen - Svein Linge)
'''



```
