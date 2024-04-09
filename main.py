import time
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import torchvision
import torch.nn as nn
from utils import window_capture
import matplotlib.pyplot as plt
from draw import drawcircle


class ChessGame:
    def __init__(self, model_path) -> None:
        self.staus = [[0]*8 for _ in range(8)]
        self.model = self.init_model(model_path)
        self.dangerous = [[0]*8 for _ in range(8)]
        self.transform=transforms.Compose([
            #transforms.GaussianBlur(5),
            transforms.Resize((100,100)),
            transforms.ToTensor()
        ])
    def get_attack_position(self, i, j, ctype):
        attack_position = []

        def attack_0(i, j):
            res = [(i+1, j-1), (i+1, j+1)]
            return res

        def attack_1(i, j):
            res = []
            ori_i = i
            ori_j = j

            i = ori_i+1
            j = ori_j+1
            while i >= 0 and j >= 0 and i <= 7 and j <= 7 and self.staus[i][j] in [6,7,8]:
                res.append((i, j))
                i += 1
                j += 1

            i = ori_i-1
            j = ori_j-1
            while i >= 0 and j >= 0 and i <= 7 and j <= 7 and self.staus[i][j] in [6,7,8]:
                res.append((i, j))
                i -= 1
                j -= 1

            i = ori_i+1
            j = ori_j-1
            while i >= 0 and j >= 0 and i <= 7 and j <= 7 and self.staus[i][j] in [6,7,8]:
                res.append((i, j))
                i += 1
                j -= 1

            i = ori_i-1
            j = ori_j+1
            while i >= 0 and j >= 0 and i <= 7 and j <= 7 and self.staus[i][j] in [6,7,8]:
                res.append((i, j))
                i -= 1
                j += 1
            return res

        def attack_2(i, j):
            res = [(i+2, j+1), (i+2, j-1), (i+1, j+2), (i+1, j-2),
                   (i-1, j-2), (i-1, j+2), (i-2, j-1), (i-2, j+1)]
            return res

        def attack_3(i, j):
            
            res = []
            ori_i = i
            ori_j = j

            i = ori_i+1
            j = ori_j
            while i >= 0 and j >= 0 and i <= 7 and j <= 7 and self.staus[i][j] in [6,7,8]:
                res.append((i, j))
                i += 1

            i = ori_i-1
            j = ori_j
            while i >= 0 and j >= 0 and i <= 7 and j <= 7 and self.staus[i][j] in [6,7,8]:
                res.append((i, j))
                i -= 1

            i = ori_i
            j = ori_j+1
            while i >= 0 and j >= 0 and i <= 7 and j <= 7 and self.staus[i][j] in [6,7,8]:
                res.append((i, j))
                j += 1

            i = ori_i
            j = ori_j-1
            while i >= 0 and j >= 0 and i <= 7 and j <= 7 and self.staus[i][j] in [6,7,8]:
                res.append((i, j))
                j -= 1
            
            return res

        def attack_4(i, j):
            res = [(i+1, j), (i+1, j-1), (i+1, j+1), (i, j-1),
                   (i, j+1), (i-1, j-1), (i-1, j), (i-1, j+1)]
            return res

        def attack_5(i, j):
            res = []
            res1 = attack_1(i, j)
            res3 = attack_3(i, j)
            res4 = attack_4(i, j)
            res.extend(res1)
            res.extend(res3)
            res.extend(res4)
            return res
        if ctype == 0:
            attack_position = attack_0(i, j)
        elif ctype == 1:
            attack_position = attack_1(i, j)
        elif ctype == 2:
            attack_position = attack_2(i, j)
        elif ctype == 3:
            attack_position = attack_3(i, j)
        elif ctype == 4:
            attack_position = attack_4(i, j)
        elif ctype == 5:
            attack_position = attack_5(i, j)

        remove_item = []
        for item in attack_position:
            if item[0] < 0 or item[0] > 7 or item[1] < 0 or item[1] > 7:    
                remove_item.append(item)
        for item in remove_item:
            attack_position.remove(item)

        return attack_position

    def init_model(self, model_path):
        resnet = torchvision.models.resnet50()
        num_classes = 9
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
        resnet.load_state_dict(torch.load(model_path))
        resnet.to('cuda')
        return resnet

    @torch.no_grad()
    def recognize(self, board):
        for i in range(8):
            for j in range(8):
                img = board.crop(
                    (j*103, i*103, min((j+1)*103, 820), min((i+1)*103, 820)
                     ))
                img_input=self.transform(img).unsqueeze(0).to('cuda')
    
                output = self.model(img_input).argmax()

                self.staus[i][j] = output.item()
               

    def show_staus(self):
        print(self.staus)

    def pd_dangerous(self):
        for i in range(8):
            for j in range(8):
                self.dangerous[i][j] = 0
        for i in range(8):
            for j in range(8):
                if self.staus[i][j] <= 5:
                    attack_position_list = self.get_attack_position(
                        i, j, self.staus[i][j])
                    for item in attack_position_list:
                        self.dangerous[item[0]][item[1]] = 1

    def draw(self):
        start_time = time.time()
        while time.time()-start_time <= 1:
            for i in range(8):
                for j in range(8):
                    if self.dangerous[i][j] == 1 and self.staus[i][j] in [6,7,8]:
                        drawcircle(550+j*102+52, 150+i*102+52, 5)


game = ChessGame('model.pth')
x = 150
y = 550

while 1:
    board = window_capture().crop((y, x, y+820, x+820))
    game.recognize(board)
    game.show_staus()
    game.pd_dangerous()
    game.draw()
    # break

'''
while 1:
    x=150
    y=550
    board=window_capture().crop((y,x,y+820,x+820))
    game.recognize(board)
'''
