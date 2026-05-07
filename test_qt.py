import sys
print("1. Python baslatildi", flush=True)
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt, QTimer
print("2. PyQt5 import OK", flush=True)
app = QApplication(sys.argv)
print("3. QApplication OK", flush=True)
w = QWidget()
w.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
w.setAttribute(Qt.WA_TranslucentBackground)
w.resize(200, 100)
w.show()
print("4. Pencere gosterildi", flush=True)
QTimer.singleShot(3000, app.quit)
sys.exit(app.exec_())
