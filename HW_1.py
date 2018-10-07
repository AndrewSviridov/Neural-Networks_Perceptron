import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtCore, QtWidgets,QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel,QMainWindow, QTableWidget, QTableWidgetItem, QWidget
from PyQt5.QtWidgets import QFormLayout, QDockWidget, QComboBox, QHBoxLayout, QPushButton, QTextEdit, QAction, QApplication, QDesktopWidget
from PyQt5.QtGui import QIcon
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import sys
import numpy as np
import random

class My_Main_window(QtWidgets.QDialog):
    def __init__(self,parent=None):
        
        super(My_Main_window,self).__init__(parent)

        # set the User Interface
        self.setWindowTitle('NN')
        self.resize(1000, 560)

        # set the figure (left&right)
        self.figure_1 = Figure(figsize=(4, 4), dpi=100)
        self.figure_2 = Figure(figsize=(4, 4), dpi=100)
        self.canvas_1 = FigureCanvas(self.figure_1)
        self.canvas_2 = FigureCanvas(self.figure_2)

        # draw the initial axes of left graph
        self.ax_1 = self.figure_1.add_axes([0.1,0.1,0.8,0.8])
        self.ax_1.set_xlim([-5,5])
        self.ax_1.set_ylim([-5,5])
        self.ax_1.plot()

        # draw the initial axes of right graph
        self.ax_2 = self.figure_2.add_axes([0.1,0.1,0.8,0.8])
        self.ax_2.set_xlim([-5,5])
        self.ax_2.set_ylim([-5,5])
        self.ax_2.plot()

        # set the button
        self.button_test = QPushButton("Test")
        self.button_train = QPushButton("Train")

        # set the combo box
        self.combo = QComboBox()
        self.combo.addItems(["perceptron1.txt", "perceptron2.txt", "2Ccircle1.txt",
                             "2Circle1.txt", "2Circle2.txt", "2CloseS.txt",
                             "2CloseS2.txt", "2CloseS3.txt", "2cring.txt",
                             "2CS.txt", "2Hcircle1.txt", "2ring.txt"])

        # set the left_table that show the valve & key value
        self.ltable = QTableWidget()
        self.ltable.setRowCount(3)
        self.ltable.setColumnCount(2)
        self.ltable.setColumnWidth(0,75)
        self.ltable.setColumnWidth(1,280)
        self.ltable.verticalHeader().setVisible(0)
        self.ltable.horizontalHeader().setVisible(0)
        self.ltable.setItem(0,0,QTableWidgetItem("Valve"))
        self.ltable.setItem(1,0,QTableWidgetItem("Key1"))
        self.ltable.setItem(2,0,QTableWidgetItem("Key2"))

        # set the right_table that show the train & test rate
        self.rtable = QTableWidget()
        self.rtable.setRowCount(2)
        self.rtable.setColumnCount(2)
        self.rtable.setColumnWidth(0,85)
        self.rtable.setColumnWidth(1,270)
        self.rtable.verticalHeader().setVisible(0)
        self.rtable.horizontalHeader().setVisible(0)
        self.rtable.setItem(0,0,QTableWidgetItem("Train Rate"))
        self.rtable.setItem(1,0,QTableWidgetItem("Test Rate"))

        # set the label
        self.label_learn=QLabel()
        self.label_time=QLabel()
        self.label_file=QLabel()
        self.label_learn.setText("Learn Rate")
        self.label_time.setText("Training Time")
        self.label_file.setText("File Name")

        # set the text editor
        self.editor_learn = QtWidgets.QLineEdit()
        self.editor_time = QtWidgets.QLineEdit()

        # the default of Learn Rate & Execution Time 
        self.learn=random.uniform(0,1)
        self.time=random.randint(0,500)

        # set the button trigger
        self.button_train.clicked.connect(self.trainFile)
        self.button_test.clicked.connect(self.testFile)

        # set the combobox trigger
        self.combo.activated.connect(self.setFile)
        
        # set the layout
        layout = QtWidgets.QVBoxLayout()
        layout_upper = QtWidgets.QHBoxLayout() #the upper level layout
        layout_down = QtWidgets.QHBoxLayout() #the lower level layout

        # insert the figure to upper layout
        layout_upper.addWidget(self.canvas_1)
        layout_upper.addWidget(self.canvas_2)

        # insert the label, test editor, button, combobox and table to lower layout
        form_layout = QFormLayout()
        form_layout.addRow(self.label_file,self.combo)
        form_layout.addRow(self.label_learn,self.editor_learn)
        form_layout.addRow(self.label_time,self.editor_time)
        form_layout.addRow(self.button_train,self.button_test)
        layout_down.addLayout(form_layout,2)
        layout_down.addWidget(self.ltable,3)
        layout_down.addWidget(self.rtable,3)
        
        layout.addLayout(layout_upper,42)
        layout.addLayout(layout_down,10)
 
        self.setLayout(layout)

    def setFile(self):
        self.ax_1.cla()
        self.ax_2.cla()
        # x ,y and out is to store the value of two dimension input and one dimension expect output
        self.x=[]
        self.y=[]
        self.out=[]

        # initial the train and test group
        self.x_train=[]
        self.y_train=[]
        self.x_test=[]
        self.y_test=[]
        self.out_train=[]
        self.out_test=[]

        # this is to store the two different group in train set
        self.x_train_bigger=[]
        self.y_train_bigger=[]
        self.x_train_smaller=[]
        self.y_train_smaller=[]

        # this is to store the two different group in test set
        self.x_test_bigger=[]
        self.y_test_bigger=[]
        self.x_test_smaller=[]
        self.y_test_smaller=[]

        # record the condition to distinguish the two different group
        self.low=int(100)

        # this is to record the bound of x_axis and y_axis 
        self.x_max=float(-100)
        self.x_min=float(100)
        self.y_max=float(-100)
        self.y_min=float(100)

        # record the number of input
        self.count=int(0)

        # record the train and test group
        self.count_train=int(0)
        self.count_test=int(0)

        # random generate a point that seperate train and test set
        self.start = int(0)
        
        # read the input file
        f=open(self.combo.currentText())
        while 1:
            line=f.readline()
            if line=="":
                break
            
            line=line[:len(line)].strip().split(" ")

            # determine the x axis range
            temp=float(line[0])
            if temp < self.x_min :
                self.x_min = temp
            if temp > self.x_max :
                self.x_max = temp
            self.x.append(temp)

            # determine the y axis range
            temp=float(line[1])
            if temp < self.y_min :
                self.y_min = temp
            if temp > self.y_max :
                self.y_max = temp
            self.y.append(temp)
            
            temp=int(line[2])
            if temp < self.low :
                self.low = temp
            self.out.append(temp)

            self.count+=1

        self.count_train = int(self.count * 2/3)
        self.count_test = self.count - self.count_train
        print(self.count_train,self.count_test)

        self.start = random.randint(0, self.count-1)

        # push the data into train and test set
        for i in range(self.count_test-1) :
            self.x_train.append(self.x[(3*i+self.start)%self.count])
            self.y_train.append(self.y[(3*i+self.start)%self.count])
            self.out_train.append(self.out[(3*i+self.start)%self.count])
            self.x_train.append(self.x[(3*i+self.start+1)%self.count])
            self.y_train.append(self.y[(3*i+self.start+1)%self.count])
            self.out_train.append(self.out[(3*i+self.start+1)%self.count])
            self.x_test.append(self.x[(3*i+self.start+2)%self.count])
            self.y_test.append(self.y[(3*i+self.start+2)%self.count])
            self.out_test.append(self.out[(3*i+self.start+2)%self.count])
        if self.count_test*3 != self.count :
            if self.count_test*3 - self.count == 2 :
                self.x_test.append(self.x[(self.start-1)%self.count])
                self.y_test.append(self.y[(self.start-1)%self.count])
                self.out_test.append(self.out[(self.start-1)%self.count])
            else :
                self.x_test.append(self.x[(self.start-2)%self.count])
                self.y_test.append(self.y[(self.start-2)%self.count])
                self.out_test.append(self.out[(self.start-2)%self.count])
                self.x_train.append(self.x[(self.start-1)%self.count])
                self.y_train.append(self.y[(self.start-1)%self.count])
                self.out_train.append(self.out[(self.start-1)%self.count])
        else :
            self.x_train.append(self.x[(self.start-2)%self.count])
            self.y_train.append(self.y[(self.start-2)%self.count])
            self.out_train.append(self.out[(self.start-2)%self.count])
            self.x_train.append(self.x[(self.start-1)%self.count])
            self.y_train.append(self.y[(self.start-1)%self.count])
            self.out_train.append(self.out[(self.start-1)%self.count])
            self.x_test.append(self.x[(self.start-3)%self.count])
            self.y_test.append(self.y[(self.start-3)%self.count])
            self.out_test.append(self.out[(self.start-3)%self.count])

        # draw the initial left graph
        self.ax_1.set_xlim([self.x_min-1,self.x_max+1])
        self.ax_1.set_ylim([self.y_min-1,self.y_max+1])
        self.ax_1.plot(self.x_train , self.y_train, '.')
        self.canvas_1.draw()

         # draw the initial right graph
        self.ax_2.set_xlim([self.x_min-1,self.x_max+1])
        self.ax_2.set_ylim([self.y_min-1,self.y_max+1])
        self.ax_2.plot(self.x_test , self.y_test, '.')
        self.canvas_2.draw()

        print("Num of Input:",self.count)

    # define the plotgraph mechanism
    def divideFile(self):
        # divive the train group and store them into two set train_bigger & train_smaller
        for i in range(self.count_train):
            if self.w1*self.x_train[i] + self.w2*self.y_train[i] - self.valve >= 0 : 
                self.x_train_bigger.append(self.x_train[i])
                self.y_train_bigger.append(self.y_train[i])
            else :
                self.x_train_smaller.append(self.x_train[i])
                self.y_train_smaller.append(self.y_train[i])

        # divive the test group and store them into two set test_bigger & test_smaller
        for i in range(self.count_test):
            if self.w1*self.x_test[i] + self.w2*self.y_test[i] - self.valve >= 0 : 
                self.x_test_bigger.append(self.x_test[i])
                self.y_test_bigger.append(self.y_test[i])
            else :
                self.x_test_smaller.append(self.x_test[i])
                self.y_test_smaller.append(self.y_test[i])

    # define the plotgraph mechanism
    def trainFile(self):
        # set Learn Rate and Execution Time
        self.learn=float(self.editor_learn.text())
        self.time=int(self.editor_time.text())
        print(self.learn,self.time)

        # set the number of correct distinguishment
        self.correct_train = float(0)
        
        line = np.linspace(-50, 50, 50)
        self.valve=random.uniform(-1,1)
        self.w1=random.uniform(-1,1)
        self.w2=random.uniform(-1,1)

        for i in range(self.time):
            index = i % self.count_train
            n_out=self.w1*self.x_train[index] + self.w2*self.y_train[index] - self.valve
            print(self.valve,self.w1,self.w2)
            
            if self.out_train[index]==self.low :
                if n_out>=0 :
                    self.valve=self.valve+self.learn
                    self.w1=self.w1-self.learn*self.x_train[index]
                    self.w2=self.w2-self.learn*self.y_train[index]
            else :
                if n_out<0 :
                    self.valve=self.valve-self.learn
                    self.w1=self.w1+self.learn*self.x_train[index]
                    self.w2=self.w2+self.learn*self.y_train[index]

        
        # calculate the correct rate
        for i in range(self.count_train) : 
            if self.out_train[i] > self.low : 
                if self.w1*self.x_train[i] + self.w2*self.y_train[i] - self.valve >= 0 :
                    self.correct_train += 1
            else:
                if self.w1*self.x_train[i] + self.w2*self.y_train[i] - self.valve < 0 :
                    self.correct_train += 1
        self.correct_train /= self.count_train

        # divide the train and test group into different set bigger&smaller
        self.divideFile()

        # print the valve & key value on left_table
        self.ltable.setItem(0,1,QTableWidgetItem(str(self.valve)))
        self.ltable.setItem(1,1,QTableWidgetItem(str(self.w1)))
        self.ltable.setItem(2,1,QTableWidgetItem(str(self.w2)))

        # print the train rate on right_table
        self.rtable.setItem(0,1,QTableWidgetItem(str(self.correct_train)))

        # draw the graph
        self.ax_1.set_xlim([self.x_min-1,self.x_max+1])
        self.ax_1.set_ylim([self.y_min-1,self.y_max+1])
        self.ax_1.plot(self.x_train_bigger , self.y_train_bigger, 'b.',
                    self.x_train_smaller , self.y_train_smaller, 'y.',
                    line, (self.valve - self.w1*line) / self.w2,'r')
        
        self.canvas_1.draw()
        self.ax_1.cla()

        print("Done")

    def testFile(self): 
        # set the number of correct distinguishment
        self.correct_test = float(0)

        # calculate the correct rate
        for i in range(self.count_test) : 
            if self.out_test[i] > self.low : 
                if self.w1*self.x_test[i] + self.w2*self.y_test[i] - self.valve >= 0 :
                     self.correct_test += 1
            else:
                if self.w1*self.x_test[i] + self.w2*self.y_test[i] - self.valve < 0 :
                     self.correct_test += 1
        self.correct_test /= self.count_test

        # print the test rate on right_table
        self.rtable.setItem(1,1,QTableWidgetItem(str(self.correct_test)))

        # draw the graph
        line = np.linspace(-50, 50, 50)
        self.ax_2.set_xlim([self.x_min-1,self.x_max+1])
        self.ax_2.set_ylim([self.y_min-1,self.y_max+1])
        self.ax_2.plot(self.x_test_bigger , self.y_test_bigger, 'b.',
                    self.x_test_smaller , self.y_test_smaller, 'y.',
                    line, (self.valve - self.w1*line) / self.w2,'r')
        
        self.canvas_2.draw()
        self.ax_2.cla()

        print("Done")

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = My_Main_window()
    main_window.show()
    app.exec()
