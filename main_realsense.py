"""
-----------Visual servo algorithm for bolt positioning in contact network-----------
Overall algorithm framework:
1. Image acquisition
2. Image processing, feature extraction (obtaining pixel coordinates)
3. Calculating image error
4. Calculating the image Jacobian matrix and its inverse matrix
5. Designing a controller to calculate the required speed of the camera
6. Update the camera pose and robot pose (i.e. calculate the transformation matrix of the robot in the next pose)
7. Solve the joint angle corresponding to the robot transformation matrix based on inverse kinematics
8. Transmit the joint angle to the robot controller (transmitting angle value or incremental value depends on the requirements of the on-site robot controller)
"""
import argparse
import os
import sys
import cv2
import socket
import math
import numpy as np
import pandas as pd
import roboticstoolbox as rtb
import machinevisiontoolbox as mvt
import torch
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtSerialPort import QSerialPort, QSerialPortInfo
from ultralytics.utils.torch_utils import select_device
from screw_detect_yolov5.detector_yolov5 import *
import pyrealsense2 as rs
from screw_detect_yolov5 import *
from App import *
from screw_detect_yolov5.models.experimental import attempt_load
from util import *
from screw_detect_yolov5.models.yolo import *
from screw_detect_yolov5.subfunction_ import *
import torch.backends.cudnn as cudnn


class MainWindow(QMainWindow, Ui_MainWindow):
    """APP interface"""
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.initialUi()
        self.signalSlot()

    def initialUi(self):
        """Interface initialization"""
        self.setMaximumSize(1127, 755)
        self.setMinimumSize(1127, 755)
        self.timer1 = QTimer(self)
        self.timer1.timeout.connect(self.ShowImg)
        self.timer1.start(550)
        self.timer2 = QTimer(self)
        self.timer2.timeout.connect(self.ShowRealTime)
        self.timer2.start(500) # ms
        self.CommunicationState = False
        self.ReadState=False
        self.VSState = False


    def signalSlot(self):
        # pushButtons
        self.pushButton_loadimg.clicked.connect(self.loadimg)
        self.pushButton_pauseimg.clicked.connect(self.pauseimg)
        self.pushButton_closeimg.clicked.connect(self.closeimg)
        self.pushButton_modelset.clicked.connect(self.ModelLoad)
        self.pushButton_modelreset.clicked.connect(self.ModelReload)
        self.pushButton_set.clicked.connect(self.SetCommunicationPara)
        self.pushButton_link.clicked.connect(self.CommunicationConnect)
        self.pushButton_openmotor.clicked.connect(self.JointMotor)
        self.pushButton_startVS.clicked.connect(self.StartVS)
        self.pushButton_endVS.clicked.connect(self.EndVS)
        self.pushButton_send.clicked.connect(self.SendData)
        self.pushButton_clear.clicked.connect(self.ClearScene)
        self.pushButton_readdesiredimg.clicked.connect(self.ReadDesiredImg)
        self.pushButton_savedesiredimg.clicked.connect(self.SaveDesiredImg)
        self.pushButton_usedefaultimg.clicked.connect(self.UseDefaultImg)
        self.pushButton_savetoexcel.clicked.connect(self.SaveExcel)

        # checkBox
        self.checkBox_serial.clicked.connect(self.SerialEnable)
        self.checkBox_socket.clicked.connect(self.SocketEnable)

        # horizontalSlider
        self.horizontalSlider_joint1.valueChanged.connect(self.Joint1Control)
        self.horizontalSlider_joint2.valueChanged.connect(self.Joint2Control)
        self.horizontalSlider_joint3.valueChanged.connect(self.Joint3Control)
        self.horizontalSlider_joint4.valueChanged.connect(self.Joint4Control)
        self.horizontalSlider_joint5.valueChanged.connect(self.Joint5Control)
        self.horizontalSlider_joint6.valueChanged.connect(self.Joint6Control)


    def loadimg(self):
        # Parameter settings
        self.CoverParas()
        self.img_1 = cv2.imread('cache.jpg', 1)
        self.label_img.setPixmap(LabelDisplayImg(self.img_1))     #Image data conversion

    def pauseimg(self):     #Pause and Resume
        if self.pushButton_pauseimg.text() == 'Pause':
            self.timer1.stop()
            self.pushButton_pauseimg.setText("Continue")
            self.pushButton_closeimg.setEnabled(False)
            self.textBrowser_info.setText("")
        else:
            self.timer1.start(1000)
            self.pushButton_pauseimg.setText("Pause")
            self.pushButton_closeimg.setEnabled(True)

    def closeimg(self):
        self.timer1.stop()
        if self.comboBox_demo.currentIndex() == 0:
            self.my_camera.release()
        self.pushButton_startVS.setEnabled(False)
        self.pushButton_loadimg.setEnabled(True)
        self.pushButton_pauseimg.setEnabled(False)
        self.pushButton_closeimg.setEnabled(False)
        self.textBrowser_info.setText("")

    def ModelLoad(self):
        self.textBrowser_info.setText("Loading...")
        pars = argparse.ArgumentParser()
        pars.add_argument("--weights_path", type=str, default=r"The absolute path of your model",
                            help="path to weights file")
        par=pars.parse_args()
        self.model, self.names, self.colors = load_yolov5_model(par.weights_path)

        self.pushButton_modelset.setEnabled(True)
        self.pushButton_modelreset.setEnabled(True)
        self.pushButton_loadimg.setEnabled(True)
        self.textBrowser_info.setText("Model loading successfully!")

    def ModelReload(self):
        self.pushButton_modelset.setEnabled(True)
        self.pushButton_modelreset.setEnabled(False)
        self.pushButton_startVS.setEnabled(False)

    def ReadDesiredImg(self):
        self.textBrowser_info.setText("Desired Image Reading...")
        self.pushButton_readdesiredimg.setEnabled(False)
        self.pushButton_savedesiredimg.setEnabled(True)
        self.ReadState = True

    def SaveDesiredImg(self):
        self.pushButton_savedesiredimg.setEnabled(False)
        self.ReadState = False

    def UseDefaultImg(self):
        self.desired_img_coordinates = computeCoordinates(self.bbox)
        self.textBrowser_info.setText("The desired image coordinates are:\n" + str(self.desired_img_coordinates))
        print("The desired image coordinates are:\n" + str(self.desired_img_coordinates))
        self.ReadState = False

    def CoverParas(self):
        self.YOLOthreshold = float(self.doubleSpinBox_YOLOconf.value())    # YOLO threshold
        self.YOLOoverlap = float(self.doubleSpinBox_YOLOnms.value())       # YOLO NMS
        self.bbox_no_use=[129, 94, 173, 231]
        self.ReadState = False
        self.VSState = False
        self.FrameIdx = 0
        self.HistoryUV = []
        self.HistoryVel = []
        self.HistoryError = []
        self.i_times=0
        self.seq = math.pi / 180

        # Servo starting position
        th_1 =    [-1.16, 26.53, 44.72, 0.00, 18.75, -1.15]
        self.q0 = np.array([th_1[0] * self.seq,
                            th_1[1] * self.seq, th_1[2] * self.seq, th_1[3] * self.seq, th_1[4] * self.seq,
                            th_1[5] * self.seq])

    def ShowImg(self):
        global data_buf, cam
        global spring_boundingboxes, distance_spring_s
        start_time = time.perf_counter()

        # 1.Image reading
        print("Center, w and h of detection box",spring_boundingboxes)
        self.img = cv2.imread('cache.jpg', 1)
        self.frame = self.img.copy()
        self.frame=cv2.resize(self.frame,(416,416))
        self.image_si = 416

        # 2.Image processing, feature extraction (acquisition of pixel coordinates)
        self.CommunicationState = True
        self.bbox_expect = [200, 191, 31, 42]    #Expected position in camera field of view
        self.desired_img_coordinates = computeCoordinates(self.bbox_expect,invert=False)
        self.bboxes = spring_boundingboxes
        self.realtime_Z=distance_spring_s
        i_2 = 0
        print("Depth of box center:", self.realtime_Z)

        if self.ReadState or self.comboBox_demo.currentIndex() == 1:
            for bbox in self.bboxes:  # 左上x，左上y，w，h
                i_2 = 0
            self.current_img_coordinates=computeCoordinates(bbox,invert=False)
            pts = np.array([[self.current_img_coordinates[0][0], self.current_img_coordinates[1][0]],
                            [self.current_img_coordinates[0][1], self.current_img_coordinates[1][1]],
                            [self.current_img_coordinates[0][2], self.current_img_coordinates[1][2]],
                            [self.current_img_coordinates[0][3], self.current_img_coordinates[1][3]]], np.int32)
            cv2.polylines(self.frame, [pts], True, (0, 0, 255), 1)
            pts_d = np.array([[self.desired_img_coordinates[0][0], self.desired_img_coordinates[1][0]],
                            [self.desired_img_coordinates[0][1], self.desired_img_coordinates[1][1]],
                            [self.desired_img_coordinates[0][2], self.desired_img_coordinates[1][2]],
                            [self.desired_img_coordinates[0][3], self.desired_img_coordinates[1][3]]], np.int32)
            cv2.polylines(self.frame, [pts_d], True, (0, 255, 0), 1)
            self.label_img.setPixmap(LabelDisplayImg(self.frame))

        elif self.VSState: # visual servoing
            if not self.comboBox_controlMode.currentText() == "VisualC":
                self.textBrowser_info.setText("Visual mode is not selected!!!")
                self.pushButton_startVS.setEnabled(True)
            elif not self.CommunicationState:
                self.textBrowser_info.setText("No communication established!")
                self.pushButton_startVS.setEnabled(True)
            elif self.comboBox_controlMode.currentText() == "VisualC" and self.CommunicationState:
                self.textBrowser_info.setText("Visual servo control is working...")
                self.bboxes=spring_boundingboxes
                i_2=0
                for bbox in self.bboxes:
                    i_2= 0
                self.current_img_coordinates = computeCoordinates(bbox, invert=False)

                # 3.Calculation of Image Error
                pts = np.array([[self.current_img_coordinates[0][0], self.current_img_coordinates[1][0]],
                                [self.current_img_coordinates[0][1], self.current_img_coordinates[1][1]],
                                [self.current_img_coordinates[0][2], self.current_img_coordinates[1][2]],
                                [self.current_img_coordinates[0][3], self.current_img_coordinates[1][3]]], np.int32)
                cv2.polylines(self.frame, [pts], True, (0, 0, 255), 1)
                pts_d = np.array([[self.desired_img_coordinates[0][0], self.desired_img_coordinates[1][0]],
                                [self.desired_img_coordinates[0][1], self.desired_img_coordinates[1][1]],
                                [self.desired_img_coordinates[0][2], self.desired_img_coordinates[1][2]],
                                [self.desired_img_coordinates[0][3], self.desired_img_coordinates[1][3]]], np.int32)
                cv2.polylines(self.frame, [pts_d], True, (0, 255, 0), 1)
                error = self.current_img_coordinates - self.desired_img_coordinates  # dim 2x4
                error = error.ravel(order='F')
                self.error = error[:, np.newaxis]
                norm_threshold = 20
                self.i_times = self.i_times + 1
                print("Convergence times：", self.i_times)
                error_norm = np.linalg.norm(self.error)
                print("Error Norm:", error_norm)
                if error_norm < norm_threshold:
                    self.timer1.stop()
                    self.end_sigal = 'EXIT'
                    self.textEdit_send.clear()
                    self.textEdit_send.setText(self.end_sigal)
                    self.SendData()
                    self.CommunicationState = False  # End communication
                    self.textBrowser_info.setText("Robot has already moved to the desired pose.")

                # 4.Calculation of Image Jacobian Matrix and Its Inverse Matrix
                cam_focal = 0.0018   # Camera focal length 600*3um=1800um
                u0v0 = [320, 240]
                rho = [3e-6, 3e-6]
                uv = np.mat(self.current_img_coordinates)
                desire_Z = 0.22    # m

                if ((self.realtime_Z[0])/1000 )>= desire_Z:
                    Z = (self.realtime_Z[0])/1000
                else:
                    Z = desire_Z
                print("Real time depth:", self.realtime_Z[0]/1000)

                # Calculation of Image Jacobian Matrix Based on Point Features
                J_image = visjac_p(cam_focal, u0v0, rho, uv, Z )  # dim 8x6 mat
                J_image_pinv = np.linalg.pinv(J_image)  # dim 6x8  mat

                # 5.Calculation of required camera speed

                lambda1 = 0.018
                cam_vel = -lambda1 * np.dot(J_image_pinv, self.error)  # dim 6x1 mat
                self.cam_vel = np.array(cam_vel)

                # 6.Calculation of the transformation matrix of the robot in the next pose

                # Construction of the ABB robot

                abb = rtb.models.DH.IRB4600()
                base2end_mat = abb.fkine(self.q0)
                end2cam_mat = mvt.SE3.Rz(math.pi/2) * mvt.SE3(0.05, 0, 0.07)
                self.cam_pose = base2end_mat * end2cam_mat
                J_robot = abb.jacobe(self.q0)
                J_robot_inv = np.linalg.inv(J_robot)

                # 7.Solve the joint angles corresponding to the robot transformation matrix based on inverse kinematics  ikine

                self.dtheta = -lambda1 * np.dot(J_robot_inv, np.dot(J_image_pinv, self.error))
                self.q0_column = self.q0.reshape(6, 1) + self.dtheta
                self.q0 = self.q0_column.reshape(1, 6).flatten()   #

                # 8.Transfer robot motion angle + save data

                self.SENDDATA()
                self.SaveData()

            else:
                self.textBrowser_info.setText("Visual servo control has been turned off.")
        else:
            pass
        stop_time = time.perf_counter()
        fps = round(1 / (stop_time - start_time), 2)
        self.label_frame.setText(str(fps))
        self.label_img.setPixmap(LabelDisplayImg(self.frame))
        self.label_img.setScaledContents(True)

    def SENDDATA(self):
        self.q_degree = [rad*180/math.pi for rad in list(self.q0)]
        self.q = list(np.round(np.array(self.q_degree), 3))
        self.data = str(self.q)
        self.textEdit_send.clear()
        self.textEdit_send.setText(self.data)
        self.SendData()

    def StartVS(self):
        self.VSState = True
        self.pushButton_startVS.setEnabled(False)
        self.pushButton_endVS.setEnabled(True)
        self.pushButton_openmotor.setEnabled(False)

    def EndVS(self):
        self.VSState = False
        self.pushButton_startVS.setEnabled(False)
        self.pushButton_endVS.setEnabled(False)

    def ShowRealTime(self):
        self.label_time.setText(time.strftime("%H:%M:%S", time.localtime()))

    def SaveData(self):
        # self.CurrentUV = self.CurrentUV.append(self.current_img_coordinates)
        currentUV = self.current_img_coordinates.ravel(order='F')
        currentUV = currentUV[:, np.newaxis]
        self.HistoryUV.append(currentUV)
        self.HistoryVel.append(self.cam_vel)   # 6x1 array
        self.HistoryError.append(self.error)   # 8x1 array

    def SaveExcel(self):
        # Save the coordinates of feature point images
        HistoryUV = np.array(self.HistoryUV).flatten()
        image_coordinates = pd.DataFrame(HistoryUV)
        writer1 = pd.ExcelWriter('ImageCoordinates.xlsx')
        image_coordinates.to_excel(writer1, 'sheet_1')
        writer1.save()

        # Save camera speed
        HistoryVel = np.array(self.HistoryVel).flatten()
        camera_vel = pd.DataFrame(HistoryVel)
        writer2 = pd.ExcelWriter('CameraVelocity.xlsx')
        camera_vel.to_excel(writer2, 'sheet_1')
        writer2.save()

        # Save feature errors
        HistoryError = np.array(self.HistoryError).flatten()
        feature_error = pd.DataFrame(HistoryError)
        writer3 = pd.ExcelWriter('FeatureError.xlsx')
        feature_error.to_excel(writer3, 'sheet_1')
        writer3.save()

    def SocketEnable(self):
        # Socket communication
        if self.checkBox_socket.isChecked():
            self.checkBox_serial.setEnabled(False)
            self.comboBox_com.setEnabled(False)
            self.comboBox_baud.setEnabled(False)
            self.comboBox_paritybit.setEnabled(False)
            self.comboBox_databit.setEnabled(False)
            self.comboBox_stopbit.setEnabled(False)
            self.lineEdit_ip.setEnabled(True)
            self.lineEdit_port.setEnabled(True)
            self.pushButton_set.setEnabled(True)
        else:
            self.checkBox_serial.setEnabled(True)
            self.lineEdit_ip.setEnabled(False)
            self.lineEdit_port.setEnabled(False)
            self.checkBox_socket.setEnabled(True)
            self.pushButton_set.setEnabled(False)
            self.pushButton_link.setEnabled(False)

    def SerialEnable(self):
        # Serial communication
        if self.checkBox_serial.isChecked():
            self.comboBox_com.setEnabled(True)
            self.comboBox_baud.setEnabled(True)
            self.comboBox_paritybit.setEnabled(True)
            self.comboBox_databit.setEnabled(True)
            self.comboBox_stopbit.setEnabled(True)
            self.pushButton_set.setEnabled(True)
            self.checkBox_socket.setEnabled(False)
        else:
            self.comboBox_com.setEnabled(False)
            self.comboBox_baud.setEnabled(False)
            self.comboBox_paritybit.setEnabled(False)
            self.comboBox_databit.setEnabled(False)
            self.comboBox_stopbit.setEnabled(False)
            self.pushButton_link.setEnabled(False)
            self.pushButton_set.setEnabled(False)
            self.checkBox_socket.setEnabled(True)

    def SetCommunicationPara(self):
        # Socket parameter settings
        ###################
        if self.checkBox_socket.isChecked():
            self.IP = self.lineEdit_ip.text()
            self.PORT = int(self.lineEdit_port.text())
            string1 = self.lineEdit_ip.text() + "\n" + self.lineEdit_port.text()
            string1 += "\nIP and Port number set successful!!!"
            self.text = self.textBrowser_info.setText(string1)
            self.textEdit_send.clear()
            self.pushButton_set.setEnabled(False)
            self.pushButton_link.setEnabled(True)
            ###################
        else:
            # Serial parameter settings
            qstr = ""
            self.serial = QSerialPort(self)
            self.serial.readyRead.connect(self.SerialReadData)
            # Set port name
            if self.comboBox_com.currentIndex() == 0:
                self.serial.setPortName("COM1")
                qstr += "Port: com1\n"
            elif self.comboBox_com.currentIndex() == 1:
                self.serial.setPortName("COM2")
                qstr += "Port: com2\n"
            elif self.comboBox_com.currentIndex() == 2:
                self.serial.setPortName("COM3")
                qstr += "Port: com3\n"
            elif self.comboBox_com.currentIndex() == 3:
                self.serial.setPortName("COM4")
                qstr += "Port: com4\n"
            # set baud rate
            if self.comboBox_baud.currentIndex() == 0:
                self.serial.setBaudRate(QSerialPort.Baud9600, QSerialPort.AllDirections)
                qstr += "BaudRate: 9600\n"
            elif self.comboBox_baud.currentIndex() == 1:
                self.serial.setBaudRate(QSerialPort.Baud115200, QSerialPort.AllDirections);
                qstr += "BaudRate: 115200\n"
            # Set parity check
            if self.comboBox_paritybit.currentIndex() == 0:
                self.serial.setParity(QSerialPort.NoParity)
                qstr += "Parity: no parity\n"
            elif self.comboBox_paritybit.currentIndex() == 1:
                self.serial.setParity(QSerialPort.OddParity)
                qstr += "Parity: odd parity\n"
            # set data bit
            if self.comboBox_databit.currentIndex() == 0:
                self.serial.setDataBits(QSerialPort.Data8)
                qstr += "DataBits: data8\n"
            elif self.comboBox_databit.currentIndex() == 1:
                self.serial.setDataBits(QSerialPort.Data7)
                qstr += "DataBits: data7\n"
            elif self.comboBox_databit.currentIndex() == 2:
                self.serial.setDataBits(QSerialPort.Data6)
                qstr += "DataBits: data6\n"
            elif self.comboBox_databit.currentIndex() == 3:
                self.serial.setDataBits(QSerialPort.Data5)
                qstr += "DataBits: data5\n"
            # Set stop position
            if self.comboBox_stopbit.currentIndex() == 0:
                self.serial.setStopBits(QSerialPort.OneStop)
                qstr += "StopBits: onestop\n"
            elif self.comboBox_stopbit.currentIndex() == 1:
                self.serial.setStopBits(QSerialPort.TwoStop)
                qstr += "StopBits: twostop\n"
            # set flow control
            self.serial.setFlowControl(QSerialPort.NoFlowControl)
            self.textBrowser_info.setText(qstr)
            self.pushButton_set.setEnabled(False)
            self.pushButton_link.setEnabled(True)

    def CommunicationConnect(self):
        # Establish a socket connection
        ###################
        if self.checkBox_socket.isChecked():
            if self.pushButton_link.text() == "Link":
                self.Socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.Socket.connect((self.IP, self.PORT))
                self.textBrowser_info.setText("Socket connecting successfully!")
                self.CommunicationState = True
                # 关闭设置菜单使能
                self.lineEdit_ip.setEnabled(False)
                self.lineEdit_port.setEnabled(False)
                self.pushButton_link.setText("Break")
            else:
                self.textEdit_send.setText("exit")
                self.Socket.close()
                del self.Socket
                self.pushButton_link.setText("Link")
                self.pushButton_set.setEnabled(True)
                self.pushButton_link.setEnabled(False)
                ###################
        # Establish a serial connection
        elif self.checkBox_serial.isChecked():
            if self.pushButton_link.text() == "Link":
                self.serial.open(QIODevice.ReadWrite)
                if (self.serial.isOpen()):
                    self.textBrowser_info.setText("Port opening successfully!")
                    self.CommunicationState = True
                    self.comboBox_com.setEnabled(False)
                    self.comboBox_baud.setEnabled(False)
                    self.comboBox_paritybit.setEnabled(False)
                    self.comboBox_databit.setEnabled(False)
                    self.comboBox_stopbit.setEnabled(False)
                    self.pushButton_link.setText("Break")
                else:
                    self.textBrowser_info.setText("Failed to open the port!")
                    self.pushButton_set.setEnabled(True)
                    self.pushButton_link.setEnabled(False)
            else:
                self.serial.close()
                self.serial.deleteLater()
                self.CommunicationState = False
                self.comboBox_com.setEnabled(True)
                self.comboBox_baud.setEnabled(True)
                self.comboBox_paritybit.setEnabled(True)
                self.comboBox_databit.setEnabled(True)
                self.comboBox_stopbit.setEnabled(True)
                self.pushButton_link.setText("Link")
                self.pushButton_set.setEnabled(True)
                self.pushButton_link.setEnabled(False)

    def SerialReadData(self):
        buf = self.serial.readAll()
        if not buf.isEmpty():
            self.textEdit_receive.clear()
            self.textEdit_receive.append(str(buf, encoding='utf-8'))
        buf.clear()

    def SendData(self):
        if self.CommunicationState:
            if self.checkBox_socket.isChecked():
                SendBytes = bytearray((self.textEdit_send.toPlainText()).encode('utf-8'))
                self.Socket.send(SendBytes)
        else:
            self.textBrowser_info.setText("No communication established!!!")

    def ClearScene(self):
        self.textEdit_send.clear()

    def JointMotor(self):
        if self.comboBox_controlMode.currentText() == "ManualC":  # Manual visual servo
            if self.CommunicationState:
                if self.pushButton_openmotor.text() == "MotorOn":
                    self.horizontalSlider_joint1.setEnabled(True)
                    self.horizontalSlider_joint2.setEnabled(True)
                    self.horizontalSlider_joint3.setEnabled(True)
                    self.horizontalSlider_joint4.setEnabled(True)
                    self.horizontalSlider_joint5.setEnabled(True)
                    self.horizontalSlider_joint6.setEnabled(True)
                    self.pushButton_openmotor.setText("MotorOff")
                    self.VSState = False
                    self.pushButton_startVS.setEnabled(False)
                    self.pushButton_endVS.setEnabled(False)
                    self.textBrowser_info.setText("Manual control has been turned on!")
                else:
                    self.horizontalSlider_joint1.setEnabled(False)
                    self.horizontalSlider_joint2.setEnabled(False)
                    self.horizontalSlider_joint3.setEnabled(False)
                    self.horizontalSlider_joint4.setEnabled(False)
                    self.horizontalSlider_joint5.setEnabled(False)
                    self.horizontalSlider_joint6.setEnabled(False)
                    self.pushButton_openmotor.setText("MotorOn")
                    self.pushButton_startVS.setEnabled(True)
                    self.pushButton_endVS.setEnabled(False)
                    self.textBrowser_info.setText("Manual control has been turned off!")
            else:
                self.textBrowser_info.setText("No communication established!!!")
        else:
            self.textBrowser_info.setText("Manual mode is not selected!!!")

    def Joint1Control(self):
        angle = self.horizontalSlider_joint1.value()  # Motor angle
        data = "joint1," + str(angle) + ",0"
        self.textEdit_send.clear()
        self.textEdit_send.setText(data)
        self.doubleSpinBox_joint1.setProperty("value", angle)
        self.SendData()

    def Joint2Control(self):
        angle = self.horizontalSlider_joint2.value()
        data = "joint2," + str(angle) + ",0"
        self.textEdit_send.clear()
        self.textEdit_send.setText(data)
        self.doubleSpinBox_joint2.setProperty("value", angle)
        self.SendData()

    def Joint3Control(self):
        angle = self.horizontalSlider_joint3.value()
        data = "joint3," + str(angle) + ",0"
        self.textEdit_send.clear()
        self.textEdit_send.setText(data)
        self.doubleSpinBox_joint3.setProperty("value", angle)
        self.SendData()

    def Joint4Control(self):
        angle = self.horizontalSlider_joint4.value()
        data = "joint4," + str(angle) + ",0"
        self.textEdit_send.clear()
        self.textEdit_send.setText(data)
        self.doubleSpinBox_joint4.setProperty("value", angle)
        self.SendData()

    def Joint5Control(self):
        angle = self.horizontalSlider_joint5.value()
        data = "joint5," + str(angle) + ",0"
        self.textEdit_send.clear()
        self.textEdit_send.setText(data)
        self.doubleSpinBox_joint5.setProperty("value", angle)
        self.SendData()

    def Joint6Control(self):
        angle = self.horizontalSlider_joint6.value()
        data = "joint6," + str(angle) + ",0"
        self.textEdit_send.clear()
        self.textEdit_send.setText(data)
        self.doubleSpinBox_joint6.setProperty("value", angle)
        self.SendData()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default=r'The absolute path of your model', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()


    app = QApplication(sys.argv)
    ui = MainWindow()
    ui.show()

    classes = opt.classes
    agnostic = opt.agnostic_nms
    save_conf=opt.save_conf
    agnostic_nms = opt.agnostic_nms
    augment = opt.augment
    confthres, iouthres = opt.conf_thres, opt.iou_thres

    with torch.no_grad():
        out, source, weights, view_img, save_txt, imgsz = \
            opt.save_dir, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
        webcam = source == '0' or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')

        # Initialize
        set_logging()
        device = select_device(opt.device)
        if os.path.exists(out):  # output dir
            shutil.rmtree(out)  # delete dir
        os.makedirs(out)  # make new dir
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
        if half:
            model.half()  # to FP16
        # Set Dataloader
        vid_path, vid_writer = None, None
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        # dataset = LoadStreams(source, img_size=imgsz)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
        pipeline.start(config)
        align_to_color = rs.align(rs.stream.color)
        spring_boundingboxes=[]
        distance_spring_s=[]

        while True:
            spring_boundingboxes, im0_s, distance_spring_s=Realsense_yolo_detect(augment, confthres, iouthres, agnostic_nms,
                                                                        device, half, save_conf,
                                                                        model, names, classes, colors, imgsz, pipeline,
                                                                        align_to_color)


    sys.exit(app.exec_())

