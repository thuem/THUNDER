#ifndef THUIMGRID_H
#define THUIMGRID_H
#include <QApplication>
#include <QFont>
#include <QPushButton>
#include <QWidget>
#include <QVector>
#include <QColor>
#include <vector>
#include <algorithm>
#include <string>
#include "mrc.h"
#include "func.h"
using namespace std;


class THUImGrid : public QWidget
{
    Q_OBJECT

public:
    THUImGrid(QWidget *parent = 0, int width = 800, int height = 600, float binning = 1);
    ~THUImGrid();

    bool init(int width = 800, int height = 600, float binning = 1);
	bool setStack(const char *mrcstack, char* classinfo=NULL);
	void setList(char *fn_selected="THUTMP_SelectedPar.txt");
	void setBox(int boxwidth, int boxheight, int boxstyle);
    
    void displayStack(int pagenum);
    int getCurPage();
    float getNStd();
    float getBin();
    vector<int> getSelected();
    void setNStd(float std);
    void setBin(float bin);

    void sortByResolution(bool bascending=true);
    void sortByPercentage(bool bascending=true);


protected:
    void paintEvent(QPaintEvent *event);
    void mousePressEvent (QMouseEvent * event);
    void keyPressEvent ( QKeyEvent * event );
    void resizeEvent(QResizeEvent *event);

    void drawrect(QPainter *painter, int x,int y);
    void readMRC(int pagenum);

    void CreateColorTable();
    void updateTitle();

    bool readClassInfo(char* classinfo);

   


private:
	 MRC m_stack;
	 string m_fnstack;
	 
    uchar *m_data;
    QVector<QRgb> m_colortable;
    int m_xbox;
    int m_ybox;
    int m_boxstyle; //0 rectangel  1 cycle


    QVector<int> m_list;
    string m_fselected;

    int m_curpage;
	

    //size of a single particle image
    int m_pwidth;
    int m_pheight;
    int m_nrow;
    int m_ncol;


    //size of display buffer
    int m_width;
    int m_height;
    
    //size of window
    float m_binning;
    int m_winwidth;
    int m_winheight;

    //for contrast
    float m_nstd;
    void float2uchar(uchar *dst, float *src);

	//for sort
	vector<ClassInfo> m_info;


 signals:
 	void OnTitleChanged(QString title);
};

#endif // THUIMGRID_H

