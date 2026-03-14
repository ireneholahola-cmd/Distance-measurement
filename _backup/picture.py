import os
import sys
import cv2
import run
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtGui, QtWidgets
from threading import Thread
from time import sleep
import time
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QImage
import 多目标  # 导入多目标.py文件
from PyQt5.QtGui import QFont  # 确保导入 QFont

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1280, 960)
        MainWindow.setStyleSheet("background-image: url(\"./template/Z2.png\")")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(330, 60, 600, 71))
        self.label.setAutoFillBackground(False)
        self.label.setStyleSheet("")
        self.label.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label.setStyleSheet("font-size:50px;font-weight:bold;font-family:SimHei;background:rgba(255,255,255,0);")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(65, 178, 698, 484))
        self.label_2.setStyleSheet("background:rgba(255,255,255,1);")
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        # 图像框
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(875, 185, 380, 450))
        self.label_3.setStyleSheet("background:rgba(255,255,255,1);")
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(22, 726, 782, 210))
        self.textBrowser.setStyleSheet("background:rgba(0,0,0,0);")
        self.textBrowser.setObjectName("textBrowser")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        # 选择文件
        self.pushButton.setGeometry(QtCore.QRect(900, 730, 120, 40))
        self.pushButton.setStyleSheet("background:rgba(53,142,255,1);border-radius:10px;padding:2px 4px;font-size:24px")
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        # 开始识别
        self.pushButton_2.setGeometry(QtCore.QRect(1100, 730, 120, 40))
        self.pushButton_2.setStyleSheet(
            "background:rgba(53,142,255,1);border-radius:10px;padding:2px 4px;font-size:24px")
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        # 退出系统
        self.pushButton_3.setGeometry(QtCore.QRect(1100, 850, 120, 40))
        self.pushButton_3.setStyleSheet(
            "background:rgba(53,142,255,1);border-radius:10px;padding:2px 4px;font-size:24px")
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        # 打开摄像头
        self.pushButton_4.setGeometry(QtCore.QRect(900, 850, 120, 40))
        self.pushButton_4.setStyleSheet(
            "background:rgba(53,142,255,1);border-radius:10px;padding:2px 4px;font-size:22px")
        self.pushButton_4.setObjectName("pushButton_4")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        font = QFont()
        font.setPointSize(16)  # 设置字体大小为 16 磅
        font.setFamily("宋体")  # 设置字体为宋体（可选）
        MainWindow.setWindowTitle(_translate("MainWindow", "熊耀阳毕业设计"))
        self.label.setText(_translate("MainWindow", "熊耀阳毕业设计"))
        self.label_2.setFont(font)
        self.label_2.setText(_translate("MainWindow", "请点击以添加视频"))
        self.label_3.setFont(font)
        self.label_3.setText(_translate("MainWindow", "生成动态图像"))
        self.pushButton.setText(_translate("MainWindow", "选择文件"))
        self.pushButton_2.setText(_translate("MainWindow", "开始识别"))
        self.pushButton_3.setText(_translate("MainWindow", "退出系统"))
        self.pushButton_4.setText(_translate("MainWindow", "打开摄像头"))

        # 点击文本框绑定槽事件
        self.pushButton.clicked.connect(self.openfile)
        self.pushButton_2.clicked.connect(self.click_1)
        self.pushButton_3.clicked.connect(self.handleCalc3)
        self.pushButton_4.clicked.connect(self.CatchUsbVideo)

    def generate_and_display_plot(self):
        pixmap = 多目标.simulate_with_file_data()
        self.label_3.setPixmap(pixmap.scaled(self.label_3.width(), self.label_3.height(), Qt.KeepAspectRatio))

    def openfile(self):
        global sname, filepath
        fname = QFileDialog()
        fname.setAcceptMode(QFileDialog.AcceptOpen)
        fname, _ = fname.getOpenFileName()
        if fname == '':
            return
        filepath = os.path.normpath(fname)
        sname = filepath.split(os.sep)
        ui.printf("当前选择的文件路径是：%s" % filepath)

    # 退出系统
    def handleCalc3(self):
        os._exit(0)

    def printf(self, text):
        self.textBrowser.append(text)
        self.cursor = self.textBrowser.textCursor()
        self.textBrowser.moveCursor(self.cursor.End)
        QtWidgets.QApplication.processEvents()

    class thread_fun(QThread):
        def run(self):
            run.run_detection_script()

    class thread2(QThread):
        def run(self):
            time.sleep(6)
            for i in range(10):
                time.sleep(5)
                if i < 2:
                    ui.printf("前方右侧有行人，请减速慢行！")
                if i > 2:
                    ui.printf("前方有车辆，请减速慢行！")

    # 开始识别
    def click_1(self):
        self.thread = self.thread_fun()
        self.thread.start()
        time.sleep(5)
        self.generate_and_display_plot()

    # 打开摄像头肉
    def CatchUsbVideo(self):
        ui.printf("打开摄像头")
        run.run_detection_script()
        for i in range(10):
            ui.printf("右侧有车辆")
            sleep(1)


class LoginDialog(QDialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle('欢迎登录')
        self.resize(600, 500)
        self.setFixedSize(self.width(), self.height())
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setStyleSheet("background-image: url(\"./template/Z1.png\")")

        self.frame = QFrame(self)
        self.frame.setStyleSheet("background:rgba(255,255,255,0);")
        self.frame.move(120, 200)

        self.mainLayout = QVBoxLayout(self.frame)
        self.nameEd1 = QLineEdit(self)
        self.nameEd1.setFixedSize(380, 30)
        self.nameEd1.setPlaceholderText("账号")
        font = QFont()
        font.setPointSize(14)
        self.nameEd1.setFont(font)
        op1 = QGraphicsOpacityEffect()
        op1.setOpacity(0.5)
        self.nameEd1.setGraphicsEffect(op1)
        self.nameEd1.setStyleSheet('''QLineEdit{border-radius:5px;}''')

        self.nameEd3 = QLineEdit(self)
        self.nameEd3.setPlaceholderText("密码")
        self.nameEd3.setFont(font)
        op5 = QGraphicsOpacityEffect()
        op5.setOpacity(0.5)
        self.nameEd3.setGraphicsEffect(op5)
        self.nameEd3.setStyleSheet('''QLineEdit{border-radius:5px;}''')
        self.under = QGroupBox()
        self.down = QHBoxLayout()
        self.under.setStyleSheet("QGroupBox {border: 0px solid transparent;}")
        self.btnOK = QPushButton('登录')
        op3 = QGraphicsOpacityEffect()
        op3.setOpacity(1)
        self.btnOK.setGraphicsEffect(op3)
        self.btnOK.setStyleSheet(
            '''QPushButton{background:#1E90FF;border-radius:5px;}QPushButton:hover{background:#4169E1;}\
            QPushButton{font-family:'Arial';color:#FFFFFF;font-size:24px;}''')

        self.btnCancel = QPushButton('注册')
        op4 = QGraphicsOpacityEffect()
        op4.setOpacity(1)
        self.btnCancel.setGraphicsEffect(op4)
        self.btnCancel.setStyleSheet(
            '''QPushButton{background:#1E90FF;border-radius:5px;}QPushButton:hover{background:#4169E1;}\
            QPushButton{font-family:'Arial';color:#FFFFFF;font-size:24px}''')

        self.mainLayout.addWidget(self.nameEd1)
        self.mainLayout.addWidget(self.nameEd3)
        self.down.addStretch(1)
        self.down.addWidget(self.btnOK)
        self.down.addStretch(1)
        self.down.addWidget(self.btnCancel)
        self.down.addStretch(1)
        self.under.setLayout(self.down)
        self.mainLayout.addWidget(self.under)
        self.mainLayout.setSpacing(50)

        # 绑定按钮事件
        self.btnOK.clicked.connect(self.button_enter_verify)
        self.btnCancel.clicked.connect(self.button_register_verify)

    def button_register_verify(self):
        global path1
        path1 = './user'
        if not os.path.exists(path1):
            os.makedirs(path1)
        user = self.nameEd1.text()
        pas = self.nameEd3.text()
        with open(path1 + '/' + user + '.txt', "w") as f:
            f.write(pas)
        self.nameEd1.setText("注册成功")

    def button_enter_verify(self):
        global administrator, userstext, passtext
        userstext = []
        passtext = []
        administrator = 0
        pw = 0
        path1 = './user'
        if not os.path.exists(path1):
            os.makedirs(path1)
        users = os.listdir(path1)

        for i in users:
            with open(path1 + '/' + i, "r") as f:
                userstext.append(i[:-4])
                passtext.append(f.readline())

        for i in users:
            if i[:-4] == self.nameEd1.text():
                with open(path1 + '/' + i, "r") as f:
                    if f.readline() == self.nameEd3.text():
                        if i[:5] == 'admin':
                            administrator = 1
                            self.accept()
                        else:
                            passtext.append(f.readline())
                            self.accept()
                    else:
                        self.nameEd3.setText("密码错误")
                        pw = 1
        if pw == 0:
            self.nameEd1.setText("账号错误")


if __name__ == "__main__":
    window_application = QApplication(sys.argv)
    login_ui = LoginDialog()
    if login_ui.exec_() == QDialog.Accepted:
        MainWindow = QtWidgets.QMainWindow()
        ui = Ui_MainWindow()
        ui.setupUi(MainWindow)
        MainWindow.show()
        if administrator == 1:
            ui.printf('欢迎管理员')
            for i in range(0, len(userstext)):
                ui.printf('账户' + str(i) + ':' + str(userstext[i]))
                ui.printf('密码' + str(i) + ':' + str(passtext[i]))
        else:
            ui.printf('欢迎用户使用先进驾驶辅助与预警系统！！！')

        sys.exit(window_application.exec_())