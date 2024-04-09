import win32gui
import win32api
import time
dc = win32gui.GetDC(0)
red = win32api.RGB(255, 0, 0)

def drawcircle(x,y,r):
    for i in range(x-r,x+r):
        for j in range(y-r,y+r):
            if (i-x)**2+(j-y)**2<=r**2:
                win32gui.SetPixel(dc,i,j,red)
def drawline_x(y,start_x,length):
    for x in range(start_x,start_x+length):
        win32gui.SetPixel(dc, y, x, red)
def drawline_y(start_y,x,length):
    for y in range(start_y,start_y+length):
        win32gui.SetPixel(dc, y, x, red)
'''
while 1:
    drawcircle(960,540,5)
'''    
