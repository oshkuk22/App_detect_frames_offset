# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import webbrowser
import pygame
from python_speech_features.sigproc import framesig
from pydub import AudioSegment
from scipy import signal
from PyQt5 import QtCore, uic, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap

AUDIO_N = 50000


def message_info(message):
    msg = QtWidgets.QMessageBox()
    msg.setIcon(QtWidgets.QMessageBox.Information)
    msg.setText(message)
    msg.setWindowTitle(u"Информация")
    msg.show()
    msg.exec_()


def open_file_(name_file):
    try:
        sound_file = AudioSegment.from_file(name_file)
        audio_samples = sound_file.get_array_of_samples()
        sound_file.export(os.path.join('temp', 'temp.wav'), format='wav')
        audio_samples = audio_samples / np.max(audio_samples)
        count_channels = sound_file.channels
        sample_rate = sound_file.frame_rate
        return audio_samples, count_channels, sample_rate
    except Exception as e:
        message_info(str(e))
        return None, None


def more_one_channel(audio_samples_):
    i = np.array(range(len(audio_samples_)))
    left_channel = audio_samples_[np.where(i % 2 == 0)]
    right_channel = audio_samples_[np.where(i % 2 != 1)]
    del i
    return left_channel, right_channel


def win_sinus_ogg_opus(length_window):
    sinus = np.empty(length_window)
    for k in range(round(length_window)):
        sinus[k] = np.sin((np.pi / 2) * np.sin((np.pi / length_window) * (k + 0.5)) ** 2)
    return sinus


def kbd_sin(length_window, alpha_, type_win='right_sin'):
    win_kaiser = signal.kaiser(length_window // 2 + 1, alpha_)
    kbd_left = np.empty(length_window // 2)
    kbd_right = np.empty(length_window // 2)
    enumerator = np.sum(win_kaiser)
    for i_ in range(length_window // 2):
        kbd_left[i_] = np.sqrt(np.sum(win_kaiser[:i_ + 1]) / enumerator)
        kbd_right[i_] = np.sqrt(np.sum(win_kaiser[:length_window - ((length_window // 2 + i_))]) / enumerator)
    w_sin_right = win_sinus(length_window)
    if type_win == 'right_sin':
        return np.hstack((kbd_left, w_sin_right[length_window // 2: length_window]))
    elif type_win == 'left_sin':
        return np.hstack((w_sin_right[: length_window // 2], kbd_right))


def kbd(length_window, alpha_):
    kbd_ = np.zeros(length_window)
    kaiser_win = signal.kaiser(length_window // 2 + 1, alpha_)
    csum = np.cumsum(kaiser_win)
    halfw = np.sqrt(csum[:-1] / csum[-1])
    kbd_[:length_window // 2] = halfw
    kbd_[-length_window // 2:] = halfw[::-1]
    return kbd_


def win_sinus(length_window):
    sinus = np.empty(length_window)
    for k in range(round(length_window)):
        sinus[k] = np.sin((np.pi / length_window) * (k + 0.5))
    return sinus


def cos_part(length_window):
    part_cos_arr = np.empty((round(length_window / 2), length_window))
    for k in range(round(length_window / 2)):
        for n in range(length_window):
            part_cos = np.cos((np.pi / (length_window / 2)) * (n + ((length_window / 2 + 1) / 2)) * (k + 0.5))
            part_cos_arr[k, n] = part_cos
    return part_cos_arr.T


def frames_signal(audio_samples, length_window, win):
    lst = []
    len_frames = (audio_samples.shape[0] - length_window) + 1
    if len_frames < 0:
        for i in range(abs(len_frames) + 1):
            audio_samples = np.append(audio_samples, 0)
        frame = audio_samples * win
        lst.append(frame)
    else:
        for elem in range(len_frames):
            frame = audio_samples[elem:(elem + length_window)] * win
            lst.append(frame)
    return lst


class DetectFramesOffset(QtWidgets.QWidget):
    def __init__(self):
        super(DetectFramesOffset, self).__init__()
        uic.loadUi(os.path.join(os.path.abspath(os.curdir), 'form', 'main.ui'), self)
        self.setWindowFlags(QtCore.Qt.WindowMinimizeButtonHint)
        self.setWindowFlags(QtCore.Qt.MSWindowsFixedSizeDialogHint)
        self.line_filename.setText(u'Аудиофайл не выбран')
        self.visible_element()
        self.progressBar_calc.setVisible(False)
        self.label_channel.setVisible(False)
        self.calculate_stop.setVisible(False)
        self.label_channel_cut.setVisible(False)
        self.play_audio.setVisible(False)
        self.stop_audio.setVisible(False)
        self.oscilo_audio.setVisible(False)
        self.flag_stop = False

        for i in range(0, 50000, 1):
            self.comboBox_shift.addItem(str(i))

        if os.path.isfile(os.path.join(os.path.abspath(os.curdir), 'temp', 'directory.txt')) and \
                os.path.getsize(os.path.join(os.path.abspath(os.curdir), 'temp', 'directory.txt')) > 0:
            with open(os.path.join(os.path.abspath(os.curdir), 'temp', 'directory.txt'), 'r') as text_file:
                self.directory = text_file.read()
        else:
            try:
                if sys.platform == 'linux':
                    self.directory = '/home'
                else:
                    self.directory = 'C:\\'
            except Exception as e:
                message_info(str(e))

        self.filename_and_dir = ''
        self.filename_audio = ''

        self.openfile.clicked.connect(self.open_file)
        self.comboBox_format.activated.connect(self.choice_combobox)
        self.comboBox_type_window.activated.connect(self.choice_combobox)
        self.comboBox_size_window.activated.connect(self.choice_combobox)
        self.calculate_detect.clicked.connect(self.analysis_frame_offset)
        self.calculate_stop.clicked.connect(self.stop_analysis)
        self.button_about_method.clicked.connect(self.about_method)
        self.play_audio.clicked.connect(self.play_audio_func)
        self.stop_audio.clicked.connect(self.stop_audio_func)
        self.oscilo_audio.clicked.connect(self.oscilo_audio_func)

    def oscilo_audio_func(self):
        plt.figure()
        plt.gcf().canvas.manager.set_window_title(self.filename_audio)
        plt.title('Осцилограмма файла ' + self.filename_audio)
        plt.xlabel('Отсчеты сигнала')
        plt.ylabel('Амплитута')
        plt.plot(self.audio_samples_)
        plt.show()

    def play_audio_func(self):
        self.set_dis(True)
        try:
            pygame.init()
            self.s = pygame.mixer.Sound(os.path.join('temp', 'temp.wav'))
            qApp.processEvents()
            self.s.play()
            self.set_dis(False)
        except Exception as e:
            self.set_dis(False)
            message_info(str(e))

    def stop_audio_func(self):
        try:
            self.s.stop()
        except Exception as e:
            message_info(str(e))

        self.set_dis(False)

    @staticmethod
    def about_method():
        try:
            webbrowser.open(os.path.realpath(os.path.join("paper", "paper.pdf")))
            # os.system(f'start {os.path.realpath(os.path.join("paper", "paper.pdf"))}')
        except Exception as e:
            message = u'Установите программу для чтения PDF файлов'
            message_info(message + 'Ошибка ' + str(e))

    def stop_analysis(self):
        self.flag_stop = True

    def set_dis(self, flag_=True):
        self.line_filename.setDisabled(flag_)
        self.openfile.setDisabled(flag_)
        self.label_format.setDisabled(flag_)
        self.comboBox_format.setDisabled(flag_)
        self.label_type_windows.setDisabled(flag_)
        self.comboBox_type_window.setDisabled(flag_)
        self.label_size_window.setDisabled(flag_)
        self.comboBox_size_window.setDisabled(flag_)
        self.label_count_sample.setDisabled(flag_)
        self.comboBox_count_sample.setDisabled(flag_)
        self.label_shift.setDisabled(flag_)
        self.comboBox_shift.setDisabled(flag_)
        self.calculate_detect.setDisabled(flag_)
        self.label_window.setDisabled(flag_)
        self.calculate_detect.setDisabled(flag_)
        self.label_window.setDisabled(flag_)
        self.label_channel.setDisabled(flag_)
        # self.progressBar_calc.setDisabled(flag_)

    def calc_frames_offset(self, audio_samples_, parts_cos_arr, window, size_window_len, frame_audio):
        self.progressBar_calc.setValue(0)
        self.progressBar_calc.setVisible(True)
        self.calculate_stop.setVisible(True)
        self.set_dis(True)
        mk_all = []

        if audio_samples_.shape[0] <= frame_audio:
            mdct = frames_signal(np.float16(audio_samples_), size_window_len, window)
            mdct = np.vstack(mdct)
            mdct = (4 / size_window_len) * (np.dot(mdct, parts_cos_arr))
            mk = (10 * np.log10(np.maximum((mdct ** 2) * (10 ** 10), 1)))
            no_null = np.zeros(audio_samples_.shape[0] - (size_window_len - 1))
            for i in range(mk.shape[0]):
                x = np.where(mk[i, :] == 0)
                no_null[i] = np.int16(mk[i, x].shape[1])
            mk_all.append(no_null)
            self.progressBar_calc.setValue(50)
            del mdct
            del mk
            del no_null
        else:
            if audio_samples_.shape[0] % frame_audio != 0:
                count = audio_samples_.shape[0] // frame_audio + 1
            else:
                count = audio_samples_.shape[0] // frame_audio

            for i in range(count):
                qApp.processEvents()
                self.progressBar_calc.setValue(int((i * 100) // count))
                if self.flag_stop:
                    message_info('Вы остановили вычисления')
                    self.calculate_stop.setVisible(False)
                    self.progressBar_calc.setValue(0)
                    self.progressBar_calc.setVisible(False)
                    self.set_dis(False)
                    break
                else:
                    if i == (count-1):

                        audio_frame = (np.float16(audio_samples_[i * frame_audio:]))

                        mdct = frames_signal(audio_frame, size_window_len, window)
                        mdct = np.vstack(mdct)
                        mdct = (4 / size_window_len) * (np.dot(mdct, parts_cos_arr))
                        mk = (10 * np.log10(np.maximum((mdct ** 2) * (10 ** 10), 1)))
                        no_null = np.zeros(audio_samples_[i * frame_audio:].shape[0])

                        for ind in range(mk.shape[0]):
                            x = np.where(mk[ind, :] == 0)
                            no_null[ind] = np.int16(mk[ind, x].shape[1])

                        mk_all.append(no_null)

                        del audio_frame
                        del mdct
                        del mk
                        del no_null
                    else:
                        audio_frame = np.float16(audio_samples_[i * frame_audio:(i * frame_audio) +
                                                                                frame_audio + size_window_len - 1])
                        mdct = frames_signal(audio_frame, size_window_len, window)
                        mdct = (np.vstack(mdct))
                        mdct = (4 / size_window_len) * (np.dot(mdct, parts_cos_arr))
                        mk = (10 * np.log10(np.maximum((mdct ** 2) * (10 ** 10), 1)))
                        no_null = np.zeros(audio_frame.shape[0] - (size_window_len - 1))
                        for i_ in range(mk.shape[0]):
                            x = np.where(mk[i_, :] == 0)
                            no_null[i_] = np.int16(mk[i_, x].shape[1])
                        mk_all.append(no_null)
                        del audio_frame
                        del mdct
                        del mk
                        del no_null
        self.progressBar_calc.setValue(100)
        self.set_dis(False)
        return mk_all

    def draw_window(self):
        plt.figure(facecolor="#F1EFEF")
        plt.title('Взвешивающее окно, используемое при анализе')
        plt.plot(self.window)
        plt.xlabel('Отсчеты')
        plt.ylabel('Значения')
        plt.savefig(os.path.join('temp', self.filename_audio.split('.')[0] + '.png'))
        plt.close('all')
        try:
            pixmap_win = QPixmap(os.path.join('temp', self.filename_audio.split('.')[0] + '.png'))
            self.label_window.setPixmap(pixmap_win)
        except Exception as e:
            message_info(str(e))

    def choice_combobox(self):
        who_is_activate = QApplication.instance().sender()
        if str(who_is_activate.objectName()) == 'comboBox_format':
            if self.comboBox_format.currentText() == 'MPEG Layer 3':
                self.comboBox_count_sample.setCurrentIndex(4)
                self.comboBox_type_window.clear()
                self.comboBox_type_window.addItem('Sine window')
                self.comboBox_type_window.setDisabled(True)
                self.comboBox_size_window.clear()
                self.comboBox_size_window.addItem('1152')
                self.comboBox_size_window.addItem('384')

                self.window = win_sinus(int(self.comboBox_size_window.currentText()))

            elif self.comboBox_format.currentText() == 'WMA':
                self.comboBox_count_sample.setCurrentIndex(4)
                self.comboBox_type_window.clear()
                self.comboBox_type_window.addItem('Sine window')
                self.comboBox_type_window.setDisabled(True)
                self.comboBox_size_window.clear()
                self.comboBox_size_window.addItems(['64', '128', '256', '512', '1024', '2048', '4096', '8192', '16384'])

                self.window = win_sinus(int(self.comboBox_size_window.currentText()))

            elif self.comboBox_format.currentText() == 'AAC':
                self.comboBox_count_sample.setCurrentIndex(4)
                self.comboBox_size_window.clear()
                self.comboBox_type_window.clear()
                self.comboBox_type_window.setDisabled(False)
                self.comboBox_type_window.addItems(['Sine window',
                                                    'Kaiser-Bessel-derived (KBD) window',
                                                    'Составное окно (слева KBD справа sine)',
                                                    'Составное окно (слева sine справа KBD)'])
                self.comboBox_size_window.clear()
                self.comboBox_size_window.addItems(['2048', '256'])

                if self.comboBox_type_window.currentText() == 'Kaiser-Bessel-derived (KBD) window':
                    if int(self.comboBox_size_window.currentText()) == 2048:
                        alpha = (4 * np.pi)
                    elif int(self.comboBox_size_window.currentText()) == 256:
                        alpha = (6 * np.pi)
                    self.window = kbd(int(self.comboBox_size_window.currentText()), alpha)
                if self.comboBox_type_window.currentText() == 'Составное окно (слева KBD справа sine)':
                    if int(self.comboBox_size_window.currentText()) == 2048:
                        alpha = (4 * np.pi)
                    elif int(self.comboBox_size_window.currentText()) == 256:
                        alpha = (6 * np.pi)
                    self.window = kbd_sin(int(self.comboBox_size_window.currentText()), alpha)
                if self.comboBox_type_window.currentText() == 'Составное окно (слева sine справа KBD)':
                    if int(self.comboBox_size_window.currentText()) == 2048:
                        alpha = (4 * np.pi)
                    elif int(self.comboBox_size_window.currentText()) == 256:
                        alpha = (6 * np.pi)
                    self.window = kbd_sin(int(self.comboBox_size_window.currentText()), alpha, type_win='left_sin')
                if self.comboBox_type_window.currentText() == 'Sine window':
                    self.window = win_sinus(int(self.comboBox_size_window.currentText()))

            self.draw_window()

        elif str(who_is_activate.objectName()) == 'comboBox_type_window':
            self.comboBox_size_window.clear()
            self.comboBox_size_window.addItems(['2048', '256'])
            if self.comboBox_type_window.currentText() == 'Kaiser-Bessel-derived (KBD) window':
                if int(self.comboBox_size_window.currentText()) == 2048:
                    alpha = (4 * np.pi)
                elif int(self.comboBox_size_window.currentText()) == 256:
                    alpha = (6 * np.pi)
                self.window = kbd(int(self.comboBox_size_window.currentText()), alpha)
            if self.comboBox_type_window.currentText() == 'Составное окно (слева KBD справа sine)':
                if int(self.comboBox_size_window.currentText()) == 2048:
                    alpha = (4 * np.pi)
                elif int(self.comboBox_size_window.currentText()) == 256:
                    alpha = (6 * np.pi)
                self.window = kbd_sin(int(self.comboBox_size_window.currentText()), alpha)
            if self.comboBox_type_window.currentText() == 'Составное окно (слева sine справа KBD)':
                if int(self.comboBox_size_window.currentText()) == 2048:
                    alpha = (4 * np.pi)
                elif int(self.comboBox_size_window.currentText()) == 256:
                    alpha = (6 * np.pi)
                self.window = kbd_sin(int(self.comboBox_size_window.currentText()), alpha, type_win='left_sin')
            if self.comboBox_type_window.currentText() == 'Sine window':
                self.window = win_sinus(int(self.comboBox_size_window.currentText()))

            self.draw_window()

        elif str(who_is_activate.objectName()) == 'comboBox_size_window':
            if self.comboBox_type_window.currentText() == 'Kaiser-Bessel-derived (KBD) window':
                if int(self.comboBox_size_window.currentText()) == 2048:
                    alpha = (4 * np.pi)
                elif int(self.comboBox_size_window.currentText()) == 256:
                    alpha = (6 * np.pi)
                self.window = kbd(int(self.comboBox_size_window.currentText()), alpha)
            elif self.comboBox_type_window.currentText() == 'Составное окно (слева KBD справа sine)':
                if int(self.comboBox_size_window.currentText()) == 2048:
                    alpha = (4 * np.pi)
                elif int(self.comboBox_size_window.currentText()) == 256:
                    alpha = (6 * np.pi)
                self.window = kbd_sin(int(self.comboBox_size_window.currentText()), alpha)
            elif self.comboBox_type_window.currentText() == 'Составное окно (слева sine справа KBD)':
                if int(self.comboBox_size_window.currentText()) == 2048:
                    alpha = (4 * np.pi)
                elif int(self.comboBox_size_window.currentText()) == 256:
                    alpha = (6 * np.pi)
                self.window = kbd_sin(int(self.comboBox_size_window.currentText()), alpha, type_win='left_sin')

            elif self.comboBox_type_window.currentText() == 'Sine window':
                        self.window = win_sinus(int(self.comboBox_size_window.currentText()))

            self.draw_window()

    def start_setting(self):
        self.comboBox_count_sample.setCurrentIndex(4)
        self.comboBox_size_window.clear()
        self.comboBox_type_window.clear()
        self.comboBox_size_window.addItem('1152')
        self.comboBox_size_window.addItem('384')
        self.comboBox_type_window.addItem('Sine window')
        self.comboBox_format.setCurrentIndex(0)


        self.window = win_sinus(int(self.comboBox_size_window.currentText()))

        plt.figure(facecolor="#F1EFEF")
        plt.title('Взвешивающее окно, используемое при анализе')
        plt.plot(self.window)
        plt.xlabel('Отсчеты')
        plt.ylabel('Значения')
        plt.savefig(os.path.join('temp', self.filename_audio.split('.')[0] + '.png'))
        plt.close('all')

        self.visible_element(flag_=True)

        try:
            pixmap_win = QPixmap(os.path.join('temp', self.filename_audio.split('.')[0] + '.png'))
            self.label_window.setPixmap(pixmap_win)
        except Exception as e:
            message_info(str(e))

    def visible_element(self, flag_=False):
        self.label_format.setVisible(flag_)
        self.comboBox_format.setVisible(flag_)
        self.label_type_windows.setVisible(flag_)
        self.comboBox_type_window.setVisible(flag_)
        self.label_size_window.setVisible(flag_)
        self.comboBox_size_window.setVisible(flag_)
        self.label_count_sample.setVisible(flag_)
        self.comboBox_count_sample.setVisible(flag_)
        self.label_shift.setVisible(flag_)
        self.comboBox_shift.setVisible(flag_)
        self.calculate_detect.setVisible(flag_)
        self.label_window.setVisible(flag_)
        self.button_about_method.setVisible(flag_)

    def open_file(self):
        self.filename_audio = ''
        filter_f = 'AUDIO (*.wav *.WAV *.Wav *.mp3 *.MP3 *.Mp3 *.wma *.WMA *.Wma *.aac *.AAC *.Acc *.mp4 *.MP4 *.Mp4)'
        options = QFileDialog.DontUseNativeDialog
        self.filename_and_dir, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Открыть аудио", self.directory,
                                                                         filter_f, options=options)
        self.line_filename.setText(self.filename_and_dir)
        if (self.filename_and_dir != u'') and (self.filename_and_dir != u'Аудиофайл не выбран'):
            self.filename_audio = self.filename_and_dir.split("/")[-1]
            self.directory = self.filename_and_dir.split(self.filename_audio)[0]
            with open(os.path.join(os.path.abspath(os.curdir), 'temp', 'directory.txt'), 'w') as text_file:
                text_file.writelines(self.directory)
        if self.filename_audio:
            self.visible_element(True)
            self.start_setting()

            self.play_audio.setVisible(True)
            self.stop_audio.setVisible(True)
            self.oscilo_audio.setVisible(True)

            self.audio_samples_, self.count_channel, self.samples_rate = open_file_(self.filename_and_dir)

            # return self.filename_and_dir
        else:
            self.play_audio.setVisible(False)
            self.stop_audio.setVisible(False)
            self.oscilo_audio.setVisible(False)
            self.line_filename.setText(u'Аудиофайл не выбран')

    def analysis_frame_offset(self):
        if self.filename_audio:
            self.label_channel.setVisible(False)
            audio_samples_2 = self.audio_samples_
            if self.count_channel == 2:
                audio_samples_ = audio_samples_2[2 * int(self.comboBox_shift.currentText()):]
                if self.comboBox_count_sample.currentText() != 'весь файл':
                    audio_samples_ = audio_samples_2[:2 * int(self.comboBox_count_sample.currentText())]
            else:
                audio_samples_ = audio_samples_2[int(self.comboBox_shift.currentText()):]
                if self.comboBox_count_sample.currentText() != 'весь файл':
                    audio_samples_ = audio_samples_2[:int(self.comboBox_count_sample.currentText())]
            part_cos_arr = cos_part(int(self.comboBox_size_window.currentText()))

            if audio_samples_ is not None:
                if self.count_channel == 2:
                    qApp.processEvents()
                    self.label_channel_cut.setVisible(True)
                    left_ch, right_ch = more_one_channel(audio_samples_)
                    self.label_channel_cut.setVisible(False)

                    self.label_channel.setVisible(True)
                    self.label_channel.setText('Вычисление окон кодирования левого канала')

                    mk_left_ = self.calc_frames_offset(left_ch, part_cos_arr, self.window,
                                                       int(self.comboBox_size_window.currentText()), AUDIO_N)

                    self.label_channel.setText('Вычисление окон кодирования правого канала')

                    if self.flag_stop:
                        self.calculate_stop.setVisible(False)
                        self.progressBar_calc.setValue(0)
                        self.progressBar_calc.setVisible(False)
                        self.set_dis(False)
                        self.label_channel.setVisible(False)
                        self.flag_stop = False
                    else:
                        mk_right_ = self.calc_frames_offset(right_ch, part_cos_arr, self.window,
                                                            int(self.comboBox_size_window.currentText()), AUDIO_N)

                        self.label_channel.setVisible(False)

                        plt.figure()
                        plt.gcf().canvas.manager.set_window_title(self.filename_audio)
                        plt.title('График нулей спектра левого канала' + ' N = ' +
                                  str(int(self.comboBox_size_window.currentText())))

                        null_ = np.hstack((mk_left_))
                        plt.plot(null_)
                        plt.xlabel('Отсчеты сигнала')
                        plt.ylabel('Количество нулей в спектре')
                        plt.show()

                        frames_signal = framesig(null_, round(int(self.comboBox_size_window.currentText()) // 2),
                                                 round(int(self.comboBox_size_window.currentText()) // 2))
                        win_offset = np.argmax(frames_signal, axis=1)
                        plt.figure()
                        plt.gcf().canvas.manager.set_window_title(self.filename_audio)
                        plt.title('График смещений окон кодирования левого канала' + ' N = ' +
                                  str(int(self.comboBox_size_window.currentText())))
                        time__l = np.empty([0])
                        for i, j in enumerate(win_offset):
                            time__l = np.append(time__l, (i * 1152) / (2 * self.samples_rate))
                        plt.plot(time__l, win_offset)
                        plt.xlabel('Время')
                        plt.ylabel('Смещение окон (отсчеты)')
                        plt.show()

                        plt.figure()
                        plt.gcf().canvas.manager.set_window_title(self.filename_audio)
                        plt.title('Гистограмма смещений окон кодирования левого канала' + ' N = ' +
                                  str(int(self.comboBox_size_window.currentText())))
                        plt.hist(win_offset, bins=int(self.comboBox_size_window.currentText()) // 2, color='blue')
                        plt.xlabel('Смещение окон (отсчеты)')
                        plt.ylabel('Частота встречаемости')
                        plt.show()

                        plt.figure()
                        plt.gcf().canvas.manager.set_window_title(self.filename_audio)
                        plt.title('График нулей спектра правого канала' + ' N = ' +
                                  str(int(self.comboBox_size_window.currentText())))
                        null_ = np.hstack(mk_right_)
                        plt.plot(null_)
                        plt.xlabel('Отсчеты сигнала')
                        plt.ylabel('Количество нулей в спектре')
                        plt.show()

                        frames_signal = framesig(null_, round(int(self.comboBox_size_window.currentText()) // 2),
                                                 round(int(self.comboBox_size_window.currentText()) // 2))
                        win_offset = np.argmax(frames_signal, axis=1)
                        plt.figure()
                        plt.gcf().canvas.manager.set_window_title(self.filename_audio)
                        plt.title('График смещений окон кодирования правого канала' + ' N = ' +
                                  str(int(self.comboBox_size_window.currentText())))
                        time__r = np.empty([0])
                        for i, j in enumerate(win_offset):
                            time__r = np.append(time__r, (i * 1152) / (2 * self.samples_rate))
                        plt.plot(time__r, win_offset)
                        plt.xlabel('Время')
                        plt.ylabel('Смещение окон (отсчеты)')
                        plt.show()

                        plt.figure()
                        plt.gcf().canvas.manager.set_window_title(self.filename_audio)
                        plt.title('Гистограмма смещений окон кодирования правого канала' + ' N = ' +
                                  str(int(self.comboBox_size_window.currentText())))
                        plt.hist(win_offset, bins=int(self.comboBox_size_window.currentText()) // 2, color='blue')
                        plt.xlabel('Смещение окон (отсчеты)')
                        plt.ylabel('Частота встречаемости')
                        plt.show()

                else:
                    if self.flag_stop:
                        message_info('Вы остановили вычисления')
                        self.progressBar_calc.setValue(0)
                        self.progressBar_calc.setVisible(False)
                        self.set_dis(False)
                    else:
                        mk_all_ = self.calc_frames_offset(audio_samples_, part_cos_arr, self.window,
                                                          int(self.comboBox_size_window.currentText()), AUDIO_N)
                        if not self.flag_stop:
                            plt.figure()
                            plt.gcf().canvas.manager.set_window_title(self.filename_audio)
                            plt.title('График нулей спектра окон кодирования' + ' N = ' +
                                      str(int(self.comboBox_size_window.currentText())))
                            null_ = np.hstack(mk_all_)
                            plt.plot(null_)
                            plt.xlabel('Отсчеты сигнала')
                            plt.ylabel('Количество нулей в спектре')
                            plt.show()

                            frames_signal = framesig(null_, round(int(self.comboBox_size_window.currentText()) // 2),
                                                     round(int(self.comboBox_size_window.currentText()) // 2))
                            win_offset = np.argmax(frames_signal, axis=1)
                            plt.figure()
                            plt.gcf().canvas.manager.set_window_title(self.filename_audio)
                            plt.title('График смещений окон кодирования' + ' N = ' +
                                      str(int(self.comboBox_size_window.currentText())))
                            time__ = np.empty([0])
                            for i, j in enumerate(win_offset):
                                time__ = np.append(time__, (i * 1152) / (2 * self.samples_rate))
                            plt.plot(time__, win_offset)
                            plt.xlabel('Время')
                            plt.ylabel('Смещение окон (отсчеты)')
                            plt.show()

                            plt.figure()
                            plt.gcf().canvas.manager.set_window_title(self.filename_audio)
                            plt.title('Гистограмма смещений окон кодирования' + ' N = ' +
                                      str(int(self.comboBox_size_window.currentText())))
                            plt.hist(win_offset, bins=int(self.comboBox_size_window.currentText()) // 2, color='blue')
                            plt.xlabel('Смещение окон (отсчеты)')
                            plt.ylabel('Частота встречаемости')
                            plt.show()
                self.flag_stop = False
            else:
                message_info('Не удается прочитать файл: ' + self.filename_audio)
        else:
            message_info('Выберите файл для анализа')

    def closeEvent(self, event):
        list_file = os.listdir(os.path.join('temp'))
        for i in list_file:
            if i != 'directory.txt':
                os.remove(os.path.join('temp', i))
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    Detect = DetectFramesOffset()
    Detect.show()
    sys.exit(app.exec_())
