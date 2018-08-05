#ifndef MAINWND_H
#define MAINWND_H
#include <QSet>
#include <QDebug>
#include <QWidget>
#include <QApplication>
//#include <QtGui/QWidget>
#include <QWidget>
#include <QComboBox>
#include <QPushButton>
#include <QLabel>
#include <QSpinBox>
#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QLineEdit>
#include <vector>
#include <string>
#include <QTextEdit>
#include <vector>
#include <QString>
#include <QScrollArea>
#include <QGroupBox>
#include <QDoubleSpinBox>
#include <QGridLayout>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QBoxLayout>
#include <QString>
#include <QSplitter>
#include <math.h>
#include <QColor>
#include <QPen>
#include <QGroupBox>
#include <stdio.h>
#include <QKeyEvent>
#include <QSize>
#include <QFileDialog>
#include <QRadioButton>
#include <QSpacerItem>
#include <QMessageBox>
#include <QScrollBar>
#include <QPalette>
#include <QHeaderView>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QMutex>
#include <QDateTime>

#include "thuimgrid.h"
#include "mrc.h"
#include <vector>

using namespace std;

class MainWnd : public QWidget
{
    Q_OBJECT

public:
    MainWnd(QWidget *parent = 0);
    ~MainWnd();

public:
    QGridLayout *m_LayoutWnd;
    THUImGrid *m_grid;;

    QPushButton *m_nextpage;
    QPushButton *m_prevpage;
    QPushButton *m_zoom;
    QDoubleSpinBox *m_bin;
    QPushButton *m_contrasti;
    QPushButton *m_contrastd;

    QGroupBox *m_groupSort;
    QComboBox *m_comboSort;
    QPushButton *m_ascend;
    QPushButton *m_descend;

	QGroupBox *m_groupSave;
    QPushButton *m_savelist;
    QPushButton *m_savepar;

public:
	string m_fnClassInfo;
	string m_fnThu;
	void setClassInfo(char *fnClassInfo);
	void setThu(char* thu);

public:
		void setZoom(float val);

		bool isInList(int classnum, vector<int> &list);

signals:
        //void xxx(int);


public slots:
    //void Onxxx();
	void OnChangeTitle(QString title);
	void OnPageUp();
	void OnPageDown();
	void OnZoom();
	void OnContrastInc();
	void OnContrastDec();
	void OnSortAscend();
	void OnSortDescend();
	void OnSaveList();
	void OnSavePar();


};

#endif // MAINWND_H


