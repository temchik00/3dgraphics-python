import numpy as np
import os
import sys
from PyQt5 import QtWidgets
import mainMenuUI

def readFromFile(filePath):
    vertTmp = []
    normalsTmp = []
    facesTmp = []
    maxPointPoly = 0
    with open(filePath, "r") as f:
        for line in f:
            parts = line.split(" ")
            if parts[0] == "v":
                tmp = []
                tmp.append(float(parts[1]))
                tmp.append(float(parts[2]) * -1)
                tmp.append(float(parts[3]))
                vertTmp.append(tmp)
            elif parts[0] == "vn":
                tmp = []
                for i in range(1, len(parts)):
                    tmp.append(float(parts[i]))
                normalsTmp.append(tmp)
            elif parts[0] == "f":
                coords = []
                for i in range(1, len(parts)):
                    tmp = parts[i].split("/")
                    coords.append(int(tmp[0])-1)
                maxPointPoly = len(coords) if len(coords) > maxPointPoly else maxPointPoly
                facesTmp.append(coords)
        f.close()
    vertices = np.full((4, len(vertTmp)), 1, dtype=np.float32)
    faces = np.full((len(facesTmp), maxPointPoly), -1, dtype=np.int64)
    for i in range(len(vertTmp)):
        for j in range(3):
            vertices[j][i] = vertTmp[i][j]
    for i in range(len(facesTmp)):
        for j in range(len(facesTmp[i])):
            faces[i][j] = facesTmp[i][j]
    return vertices, faces


class Menu(QtWidgets.QMainWindow, mainMenuUI.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.openFileDialogButton.clicked.connect(self.openFileDialog)
        self.startButton.clicked.connect(self.quit)
        self.file = None
        self.tmp = None

    def openFileDialog(self):
        name = QtWidgets.QFileDialog.getOpenFileName(self, "Open file", '', "Object files (*.obj)")[0]
        self.fileNameShow.setText("File name: " + name)
        self.tmp = name

    def quit(self):
        self.file = self.tmp
        self.close()


def showMenu():
    app = QtWidgets.QApplication(sys.argv)
    window = Menu()
    window.show()
    app.exec_()
    return window.file
