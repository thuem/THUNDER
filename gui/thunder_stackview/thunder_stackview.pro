TEMPLATE = app
TARGET = thunder_stackview
DEPENDPATH += .
QT += svg core gui widgets
INCLUDEPATH += .
LIBS +=
CONFIG += static plugin

# Input
HEADERS += MainWnd.h mrc.h func.h thuimgrid.h
SOURCES += main.cpp \
           MainWnd.cpp \
           mrc.cpp \
           func.cpp \
           thuimgrid.cpp
