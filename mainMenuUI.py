# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Python\3dgraphics-python\3dGraphics.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(324, 194)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.fileNameShow = QtWidgets.QLabel(self.centralwidget)
        self.fileNameShow.setGeometry(QtCore.QRect(10, 10, 311, 21))
        self.fileNameShow.setScaledContents(True)
        self.fileNameShow.setObjectName("fileNameShow")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(80, 120, 164, 29))
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.openFileDialogButton = QtWidgets.QPushButton(self.widget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.openFileDialogButton.setFont(font)
        self.openFileDialogButton.setObjectName("openFileDialogButton")
        self.horizontalLayout.addWidget(self.openFileDialogButton)
        self.startButton = QtWidgets.QPushButton(self.widget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.startButton.setFont(font)
        self.startButton.setObjectName("startButton")
        self.horizontalLayout.addWidget(self.startButton)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 324, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Scene render"))
        self.fileNameShow.setText(_translate("MainWindow", "File name: "))
        self.openFileDialogButton.setText(_translate("MainWindow", "Open file"))
        self.startButton.setText(_translate("MainWindow", "Start"))
