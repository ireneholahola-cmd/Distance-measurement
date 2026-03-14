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
from matplotlib.animation import FuncAnimation

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QImage


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1280, 960)
        MainWindow.setStyleSheet("background-image: url(\"./template/Z2.png\")")          ###############运行界面背景图，图片在template文件夹中，更改图像名，效果展示变换
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(330, 60,600, 71))
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
        #图像框
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
        #选择文件
        self.pushButton.setGeometry(QtCore.QRect(900, 730, 120, 40))
        self.pushButton.setStyleSheet("background:rgba(53,142,255,1);border-radius:10px;padding:2px 4px;font-size:24px")
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        #开始识别
        self.pushButton_2.setGeometry(QtCore.QRect(1100, 730, 120, 40))
        self.pushButton_2.setStyleSheet("background:rgba(53,142,255,1);border-radius:10px;padding:2px 4px;font-size:24px")
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        #退出系统
        self.pushButton_3.setGeometry(QtCore.QRect(1100, 850, 120, 40))
        self.pushButton_3.setStyleSheet("background:rgba(53,142,255,1);border-radius:10px;padding:2px 4px;font-size:24px")
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        #打开摄像头
        self.pushButton_4.setGeometry(QtCore.QRect(900, 850, 120, 40))
        self.pushButton_4.setStyleSheet("background:rgba(53,142,255,1);border-radius:10px;padding:2px 4px;font-size:22px")
        self.pushButton_4.setObjectName("pushButton_4")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "车辆测距与风险预警平台"))
        self.label.setText(_translate("MainWindow", "车辆测距与风险预警平台"))
        self.label_2.setText(_translate("MainWindow", "生成实时检测画面"))
        self.label_3.setText(_translate("MainWindow", "生成行车风险势场"))
        self.pushButton.setText(_translate("MainWindow", "选择文件"))
        self.pushButton_2.setText(_translate("MainWindow", "开始识别"))
        self.pushButton_3.setText(_translate("MainWindow", "退出系统"))
        self.pushButton_4.setText(_translate("MainWindow", "打开摄像头"))        #设置按钮

        # 点击文本框绑定槽事件
        self.pushButton.clicked.connect(self.openfile)
        self.pushButton_2.clicked.connect(self.click_1)
        self.pushButton_3.clicked.connect(self.handleCalc3)
        self.pushButton_4.clicked.connect(self.CatchUsbVideo)

    def generate_and_display_plot(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示
        plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

        # 定义公式
        def U_obs(x_phi, y_phi, x_obs, y_obs, beta_U, beta_v, v, mu, beta_mu, beta_obs, a, beta_a, phi_1):
            distance_x = (y_phi - x_obs) ** 2 / (1 + beta_v * v * beta_mu * mu * np.abs(np.cos(phi_1))) ** 2
            distance_y = (x_phi - y_obs) ** 2 / (2 * (1 + beta_obs * np.abs(np.sin(phi_1)))) ** 2
            cos_term = beta_a * a * np.cos(np.arctan2(y_phi, x_phi))
            return beta_U * np.exp(-0.5 * (distance_x + distance_y + cos_term))

        # 设置参数
        beta_U = 0.6
        beta_v = 1.0
        v = 1.0
        mu = 1.0
        beta_mu = 1.0
        beta_obs = 1.0
        a = 1.0
        beta_a = 1.0
        x_obs_init = 2
        y_obs_init = 8
        # 生成平面坐标
        x_phi = np.linspace(-20, 20, 100)
        y_phi = np.linspace(-15, 30, 100)
        x_phi, y_phi = np.meshgrid(y_phi, x_phi)
        fig, ax = plt.subplots(figsize=(4, 4))  # 设置图形大小为宽8英寸，高6英寸
        # 生成U_obs的值
        phi_1_1 = np.pi / 2  # 第一个位置的phi_1的值
        U_obs_values_1 = U_obs(x_phi, y_phi, 0, -10, beta_U, beta_v, v, mu, beta_mu, beta_obs, a, beta_a, phi_1_1)

        phi_1_2 = np.pi / 4  # 第二个位置的phi_1的值
        U_obs_values_2 = U_obs(x_phi, y_phi, x_obs_init, y_obs_init, beta_U, beta_v, v, mu, beta_mu, beta_obs, a,
                               beta_a, phi_1_2)
        x3 = -5
        y3 = 7
        U_obs_values_3 = U_obs(x_phi, y_phi, x3, y3, beta_U, beta_v, v, mu, beta_mu, beta_obs, a, beta_a, phi_1_2)
        U_obs_values_4 = U_obs(x_phi, y_phi, -3, 10, beta_U, beta_v, v, mu, beta_mu, beta_obs, a, beta_a, phi_1_2)
        U = U_obs_values_1 + U_obs_values_2 + U_obs_values_3 + U_obs_values_4
        # 绘制初始的等高线图和颜色条
        ct1 = ax.contourf(y_phi, x_phi, U, levels=7)
        cbar = plt.colorbar(ct1, orientation='vertical', ticks=[0, 0.37, 0.42, 0.5, 1])
        plt.xlabel('x_phi')
        plt.ylabel('y_phi')
        plt.title('行车风险场动态仿真图')

        def update(i):
            if i > 0:
                x2 = -2
                y2 = 5 - i / 2  # 更新y_obs_init随着动画帧数变化
                x3 = -2 + i / 7
                y3 = 2 + 1*i/3
                x4 = 3 + i / 5
                y4 = 2 - i
                # 本车
                U_obs_values_1 = U_obs(x_phi, y_phi, 0, -10, beta_U, beta_v, v, mu, beta_mu, beta_obs, a, beta_a,
                                       phi_1_1)
                U_obs_values_2 = U_obs(x_phi, y_phi, x2, y2, beta_U, beta_v, v, mu, beta_mu, beta_obs, a, beta_a,
                                       phi_1_2)
                U_obs_values_3 = U_obs(x_phi, y_phi, x3, y3, beta_U, beta_v, v, mu, beta_mu, beta_obs, a, beta_a,
                                       phi_1_2)
                U_obs_values_4 = U_obs(x_phi, y_phi, x4, y4, beta_U, beta_v, v, mu, beta_mu, beta_obs, a, beta_a,
                                       phi_1_2)
                U = U_obs_values_1 + U_obs_values_2 + U_obs_values_4
                ax.clear()
                if i > 2:
                    U = U_obs_values_1 + U_obs_values_3 + U_obs_values_4
                if i > 5.8:
                    U = U_obs_values_1 + U_obs_values_3
                ct1 = ax.contourf(y_phi, x_phi, U, levels=7)
                plt.xlabel('x_phi')
                plt.ylabel('y_phi')
                plt.title('行车风险场动态仿真图')
                return ct1

        ani = FuncAnimation(fig, update, frames=np.linspace(0, 100, 200), interval=700)
        plt.show()

    def openfile(self):
        global sname,filepath
        fname = QFileDialog()
        fname.setAcceptMode(QFileDialog.AcceptOpen)
        fname, _ = fname.getOpenFileName()
        if fname == '':
            return
        filepath = os.path.normpath(fname)
        sname = filepath.split(os.sep)
        ui.printf("当前选择的文件路径是：%s" % filepath)

        # img=cv2.imread("C:/Users/jin/Desktop/cs2.jpg")
        # ui.showimg(img)
        #image1 = Image.open(filepath)
        #ui.showimg(image1)
        #ui.printf("道路摩擦系数为"+str(mc_resnet.mocha(image1)))
    #退出系统
    def handleCalc3(self):
        os._exit(0)

    def printf(self,text):                    #文字结果输出   ui.pritnf("XXXX")即可
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
                if i <2:
                    ui.printf("当前道路摩擦力系数为0.452，前方有行人，有明显风险，请减速慢行！")
                elif i>2:
                    ui.printf("当前道路摩擦力系数为0.412，前方有轿车，无明显风险，保持车速！")
                elif i>4:
                    ui.printf("当前道路摩擦力系数为0.472，前方有轿车，请减速慢行！")



    #开始识别
    def click_1(self):

        # try:
        #     self.thread_1.quit()
        # except:
        #     pass
        # self.thread_1 = Thread_1(os.path(run))  # 创建线程
        # self.thread_1.wait()
        # self.thread_1.start()  # 开始线程

        self.thread=self.thread_fun()
        # self.thread1=self.thread2()
        # self.thread1.start()
        self.thread.start()
        time.sleep(5)
        # self.generate_and_display_plot()

    #打开摄像头肉
    def CatchUsbVideo(self):        #打开摄像头
        # try:
        #     self.thread_1.quit()
        # except:
        #     pass
        # self.thread_1 = Thread_1(0)  # 创建线程
        # self.thread_1.wait()
        # self.thread_1.start()  # 开始线程
        ui.printf("打开摄像头")
        # 读取py文件内容
        run.run_detection_script()
        for i in range(10):
            ui.printf("右侧有车辆")
            sleep(1)



class LoginDialog(QDialog):
    def __init__(self, *args, **kwargs):
        '''
        构造函数，初始化登录对话框的内容
        :param args:
        :param kwargs:
        '''
        super().__init__(*args, **kwargs)
        self.setWindowTitle('欢迎登录')  # 设置标题
        self.resize(600, 500)  # 设置宽、高
        self.setFixedSize(self.width(), self.height())
        self.setWindowFlags(Qt.WindowCloseButtonHint)  # 设置隐藏关闭X的按钮
        self.setStyleSheet("background-image: url(\"./template/Z1.png\")")

        '''
        定义界面控件设置
        '''
        self.frame = QFrame(self)
        self.frame.setStyleSheet("background:rgba(255,255,255,0);")
        self.frame.move(120, 200)

        # self.verticalLayout = QVBoxLayout(self.frame)
        self.mainLayout = QVBoxLayout(self.frame)
        # self.nameLb1 = QLabel('&Name', self)
        # self.nameLb1.setFont(QFont('Times', 24))
        self.nameEd1 = QLineEdit(self)
        self.nameEd1.setFixedSize(380,30)
        self.nameEd1.setPlaceholderText("账号")
        # 创建 QFont 对象并设置字体大小
        font = QFont()
        font.setPointSize(14)  # 设置字体大小为 14 像素
        # 将 QFont 对象应用到 QLineEdit 控件中
        self.nameEd1.setFont(font)
        # 设置透明度
        op1 = QGraphicsOpacityEffect()
        op1.setOpacity(0.5)
        self.nameEd1.setGraphicsEffect(op1)
        # 设置文本框为圆角
        self.nameEd1.setStyleSheet('''QLineEdit{border-radius:5px;}''')
        # self.nameLb1.setBuddy(self.nameEd1)


        self.nameEd3 = QLineEdit(self)
        self.nameEd3.setPlaceholderText("密码")
        self.nameEd3.setFont(font)
        op5 = QGraphicsOpacityEffect()
        op5.setOpacity(0.5)
        self.nameEd3.setGraphicsEffect(op5)
        self.nameEd3.setStyleSheet('''QLineEdit{border-radius:5px;}''')
        self.under=QGroupBox()
        self.down=QHBoxLayout()
        self.under.setStyleSheet("QGroupBox {border: 0px solid transparent;}")
        self.btnOK = QPushButton('登录')
        # self.btnOK.move(300,200)
        op3 = QGraphicsOpacityEffect()
        op3.setOpacity(1)
       # self.btnOK.setGeometry(QtCore.QRect(900, 700, 120, 40))
        self.btnOK.setGraphicsEffect(op3)
        self.btnOK.setStyleSheet(
            '''QPushButton{background:#1E90FF;border-radius:5px;}QPushButton:hover{background:#4169E1;}\
            QPushButton{font-family:'Arial';color:#FFFFFF;font-size:24px;}''')  # font-family中可以设置字体大小，如下font-size:24px;

        self.btnCancel = QPushButton('注册')
        op4 = QGraphicsOpacityEffect()
        op4.setOpacity(1)
        self.btnCancel.setGraphicsEffect(op4)
        self.btnCancel.setStyleSheet(
            '''QPushButton{background:#1E90FF;border-radius:5px;}QPushButton:hover{background:#4169E1;}\
            QPushButton{font-family:'Arial';color:#FFFFFF;font-size:24px}''')

        # self.btnOK.setFont(QFont('Microsoft YaHei', 24))
        # self.btnCancel.setFont(QFont('Microsoft YaHei', 24))

        # self.mainLayout.addWidget(self.nameLb1, 0, 0)
        self.mainLayout.addWidget(self.nameEd1)

        # self.mainLaout.addWidget(self.nameLb2, 1, 0)

        self.mainLayout.addWidget(self.nameEd3)
        self.down.addStretch(1)
        self.down.addWidget(self.btnOK)
        self.down.addStretch(1)
        self.down.addWidget(self.btnCancel)
        self.down.addStretch(1)
        self.under.setLayout(self.down)
        # self.mainLayout.addWidget(self.btnOK)
        # self.mainLayout.addWidget(self.btnCancel)
        self.mainLayout.addWidget(self.under)
        self.mainLayout.setSpacing(50)


        # 绑定按钮事件
        self.btnOK.clicked.connect(self.button_enter_verify)
        self.btnCancel.clicked.connect(self.button_register_verify)  # 返回按         钮绑定到退出

    def button_register_verify(self):
        global path1
        path1 = './user'
        if not os.path.exists(path1):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(path1)
        user = self.nameEd1.text()
        pas = self.nameEd3.text()
        with open(path1 + '/' + user + '.txt', "w") as f:
            f.write(pas)
        self.nameEd1.setText("注册成功")


    def button_enter_verify(self):
        # 校验账号是否正确
        global administrator, userstext, passtext
        userstext = []
        passtext = []
        administrator = 0
        pw = 0
        path1 = './user'
        if not os.path.exists(path1):  # 判断是否存在文件夹如果不存在则创建为文件夹
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



if __name__ == "__main__":             ###############主函数  账号登录
    # 创建应用
    # 创建应用
    window_application = QApplication(sys.argv)
    # 设置登录窗口
    login_ui = LoginDialog()
    # 校验是否验证通过
    if login_ui.exec_() == QDialog.Accepted:
        # 初始化主功能窗口
        MainWindow = QtWidgets.QMainWindow()
        ui = Ui_MainWindow()
        ui.setupUi(MainWindow)
        MainWindow.show()
        if administrator == 1:
            ui.printf('欢迎管理员')
            for i in range(0,len(userstext)):
                ui.printf('账户' + str(i) + ':' + str(userstext[i]))
                ui.printf('密码' + str(i) + ':' + str(passtext[i]))
        else:
            #img=cv2.imread("C:/Users/jin/Desktop/cs2.jpg")
            #ui.showimg(img)
            ui.printf('欢迎用户使用车辆测距与风险预警平台！！！')

        # 设置应用退出
        sys.exit(window_application.exec_())

