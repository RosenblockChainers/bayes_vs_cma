import sys, os, pickle
from PyQt4.QtCore import *
from PyQt4.QtGui import *

import matplotlib
matplotlib.use('Qt4Agg')
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib import cm
from matplotlib.lines import Line2D
import contour_function as function
import numpy as np
import pandas as pd
import sys
from solution import RealSolution

# colormap 白->青
cdict = {'red': ((0.0, 0.0, 0.0),
                 (1.0, 1.0, 1.0)),
         'green': ((0.0, 0.0, 0.0),
                   (1.0, 1.0, 1.0)),
         'blue': ((0.0, 1.0, 1.0),
                  (1.0, 1.0, 1.0)),
         'yellow': ((1.0, 1.0, 1.0),
                  (1.0, 1.0, 1.0))
         }

bw = LinearSegmentedColormap('BlueWhile', cdict)

class AppForm(QMainWindow):
    def __init__(self, log_path, func=None, parent=None):
        QMainWindow.__init__(self, parent)
        self.setWindowTitle('Viewer: %s' % log_path)

        self.n = 2 # 次元数
        self.func = func # 目的関数

        self.create_menu()
        self.create_main_frame()
        self.create_status_bar()

        self.log_path = log_path # パス
        self.df = pd.DataFrame.from_csv('%s/log_data.csv' % log_path)
        self.maxg=len(self.df) # 最大世代数
        self.g = 1 # 現世代数

        self.timer = QTimer(self) # タイマー
        self.timer.timeout.connect(self.increment_g)

        self.textbox.setText('1')
        self.init_draw()

    def increment_g(self):
        '''世代数をインクリメントする'''
        if self.g < self.maxg:
            self.g += 1
            self.textbox.setText('%d' % self.g)
            self.on_draw()
        else:
            self.timer.stop()

    def decrement_g(self):
        '''世代数をデクリメントする'''
        if self.g > 1:
            self.g -= 1
            self.textbox.setText('%d' % self.g)
            self.on_draw()

    def reset_g(self):
        self.g = 1
        self.textbox.setText('%d' % self.g)
        self.on_draw()

    def init_draw(self):
        with open('%s/logData%d.obj' % (self.log_path, self.g), 'rb') as f:
            data = pickle.load(f)

        solutions = data['X']

        self.weight = np.maximum(0, np.log(len(solutions)/2 + 1) - np.log(np.arange(1, int(len(solutions)) + 1)))
        self.points = []
        self.axes.plot(0, 0, 'y*', ms=10)

        degree = np.arange(0, 2*np.pi, 0.01)
        self.circlex = np.array([np.cos(degree), np.sin(degree)])
        for i, s in enumerate(solutions):
            point, = self.axes.plot(s[0], s[1], 'bo', alpha=0.8, ms=6*self.slider.value()/50)
            self.points.append(point)

        self.canvas.draw()

    def on_draw(self):
        '''Redraws the figure'''
        self.g = int(self.textbox.text())

        f = open('%s/logData%d.obj' % (self.log_path, self.g), 'rb')
        data = pickle.load(f)
        f.close()

        solutions = data['X']
        print("X:", solutions)

        self.points = []
        self.canvas.draw()
        for i, s in enumerate(solutions):
            point, = self.axes.plot(s[0], s[1], 'bo', alpha=0.8, ms=6*self.slider.value()/50)
            self.points.append(point)

        self.axes.grid(self.grid_cb.isChecked())

        self.canvas.draw()

    def on_start(self):
        '''タイマーをスタートする'''
        if self.radio_button_slow.isChecked():
            self.timer.start(1600)
        elif self.radio_button_default.isChecked():
            self.timer.start(800)
        else:
            self.timer.start(400)

    def on_stop(self):
        self.timer.stop()

    def restart(self):
        self.timer.stop()
        if self.radio_button_slow.isChecked():
            self.timer.start(1600)
        elif self.radio_button_default.isChecked():
            self.timer.start(800)
        else:
            self.timer.start(400)

    def create_main_frame(self):
        self.main_frame = QWidget()

        self.dpi = 72
        self.fig = Figure((7.0, 7.0), dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)

        self.axes = self.fig.add_subplot(111)
        self.axes.set_xlim(-1.2, 1.2)
        self.axes.set_ylim(-1.2, 1.2)

        aspect = (self.axes.get_xlim()[1] - self.axes.get_xlim()[0]) / (self.axes.get_ylim()[1] - self.axes.get_ylim()[0])
        self.axes.set_aspect(aspect)
        self.axes.set_xlabel(r'$x_0$')
        self.axes.set_ylabel(r'$x_1$')
        self.axes.locator_params(axis='both', nbins=4)

        if self.func is not None:
            X = np.arange(-5.5, 5.5, 0.01)
            Y = np.arange(-5.5, 5.5, 0.01)
            # X = np.arange(-20.5, 60.0, 0.01)
            # Y = np.arange(-20.5, 20.5, 0.01)
            X, Y = np.meshgrid(X, Y)
            #self.axes.contourf(X, Y, np.log(self.func(Y, X)+1), 30, cmap=bw)
            # self.axes.contour(X, Y, np.log(self.func(Y, X)+1), 30, cmap=bw)
            self.axes.contour(X, Y, np.log(self.func(X, Y)+1), 30, cmap=bw)

        self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)

        self.g_label = QLabel('g =')
        self.textbox = QLineEdit()
        self.textbox.setMinimumWidth(20)
        self.textbox.setMaximumWidth(40)
        self.connect(self.textbox, SIGNAL('editingFinished()'), self.on_draw)

        self.reset_button = QPushButton('&<<')
        self.connect(self.reset_button, SIGNAL('clicked()'), self.reset_g)

        self.prev_button = QPushButton('&<')
        self.connect(self.prev_button, SIGNAL('clicked()'), self.decrement_g)

        self.start_button = QPushButton('&Start')
        self.connect(self.start_button, SIGNAL('clicked()'), self.on_start)

        self.stop_button = QPushButton('&Stop')
        self.connect(self.stop_button, SIGNAL('clicked()'), self.on_stop)

        self.next_button = QPushButton('&>')
        self.connect(self.next_button, SIGNAL('clicked()'), self.increment_g)

        self.grid_cb = QCheckBox('Show &Grid')
        self.grid_cb.setChecked(False)
        self.connect(self.grid_cb, SIGNAL('stateChanged(int)'), self.on_draw)

        self.weight_cb = QCheckBox('Show &Weight')
        self.weight_cb.setChecked(False)
        self.connect(self.weight_cb, SIGNAL('stateChanged(int)'), self.on_draw)

        slider_label = QLabel('Point size(%):')
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(1, 100)
        self.slider.setValue(50)
        self.slider.setValue(50)
        self.slider.setTracking(True)
        self.slider.setTickPosition(QSlider.TicksBothSides)
        self.connect(self.slider, SIGNAL('valueChanged(int)'), self.on_draw)

        #self.speed_box = QGroupBox('speed')
        self.radio_button_slow = QRadioButton('slow')
        self.connect(self.radio_button_slow, SIGNAL('clicked()'), self.restart)
        self.radio_button_default = QRadioButton('default')
        self.connect(self.radio_button_default, SIGNAL('clicked()'), self.restart)
        self.radio_button_fast = QRadioButton('fast')
        self.connect(self.radio_button_fast, SIGNAL('clicked()'), self.restart)
        self.radio_button_default.setChecked(True)
        radio_button_layout = QHBoxLayout()
        radio_button_layout.addWidget(QLabel('speed:'))
        radio_button_layout.addWidget(self.radio_button_slow)
        radio_button_layout.addWidget(self.radio_button_default)
        radio_button_layout.addWidget(self.radio_button_fast)
        radio_button_layout.addStretch()
        #self.speed_box.setLayout(radio_button_layout)

        hbox = QHBoxLayout()
        hbox2 = QHBoxLayout()

        for w in [ self.g_label, self.textbox, self.reset_button, self.prev_button, self.start_button,
                   self.stop_button, self.next_button, ]:
            hbox.addWidget(w)
            hbox.setAlignment(w, Qt.AlignVCenter)
        for w in [ self.grid_cb, self.weight_cb, slider_label, self.slider ]:
            hbox2.addWidget(w)
            hbox2.setAlignment(w, Qt.AlignVCenter)

        vbox = QVBoxLayout()
        vbox.addWidget(self.canvas)
        vbox.addWidget(self.mpl_toolbar)
        vbox.addLayout(hbox)
        vbox.addLayout(hbox2)
        vbox.addLayout(radio_button_layout)

        self.main_frame.setLayout(vbox)
        self.setCentralWidget(self.main_frame)

    def create_status_bar(self):
        self.status_text = QLabel('This is a demo')
        self.statusBar().addWidget(self.status_text, 1)

    def create_menu(self):
        self.file_menu = self.menuBar().addMenu('&File')

        quit_action = self.create_action('&Quit', shortcut='Ctrl+Q', slot=self.close,
                                         tip='Close the application')

        self.add_actions(self.file_menu, (None, quit_action))

    def add_actions(self, target, actions):
        for action in actions:
            if action is None:
                target.addSeparator()
            else:
                target.addAction(action)

    def create_action(self, text, slot=None, shortcut=None, icon=None,
                      tip=None, checkable=False, signal='triggerer()'):
        action = QAction(text, self)
        if icon is not None:
            action.setIcon(QIcon(':/%s.png' % icon))
        if shortcut is not None:
            action.setShortcut(shortcut)
        if tip is not None:
            action.setToolTip(tip)
            action.setStatusTip(tip)
        if slot is not None:
            self.connect(action, SIGNAL(signal), slot)
        if checkable:
            action.setCheckable(True)

        return action


def main(log_path='./', func=None):
    app = QApplication(sys.argv)
    form = AppForm(log_path, func=func)
    form.show()
    app.exec_()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit("2d_viewer.py ログディレクトリ")
    log_path = sys.argv[1]
    main(log_path=log_path, func=function.sphere)