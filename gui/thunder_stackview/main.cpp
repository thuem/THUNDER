//#include <QtGui/QApplication>
#include <QApplication>
#include "MainWnd.h"
#include "func.h"
#include <stdlib.h>

//Input: 1input.mrc 2bin
int main(int narg, char *argc[])
{
    if(narg==1)
    {
        printf("\nTHUNDER -- Stack Viewer\n");
        printf("Version 0.0 (Jan 02, 2018)\n\n");

        printf("Input: input.mrc [Options]\n");
        printf("Options   Value(default)   Description\n");
        printf("   -bin   1                Binning factor\n");
        printf("   -cls   none             Class_Info_XXX.txt\n");
        printf("   -thu   none             Meta_Round_XXX.thu\n");
        printf("   -srt   1                -1/0/1, -1: sort descending, 0: not sort, 1: sort ascending\n");
        printf("   -srp   1                0/1, 0: sort by percentage, 1: sort by resolution\n");

        printf("\nWrote by Xueming Li @ Tsinghua University\n");

        return 0;
    }

    QApplication a(narg, argc);
    MainWnd w;

    char *arg=NULL;
    bool bsortres=true;


    int width=640;
    int height=480;

    float bin=1.0;
    arg=readOption(narg, argc, 2, "-bin");
    if(arg!=NULL) bin=atof(arg);
    w.setZoom(1.0/bin);

    w.m_grid->setMinimumSize(width,height);
    w.m_grid->init(width,height,bin);

    arg=readOption(narg, argc, 2, "-cls");
    w.m_grid->setStack(argc[1], arg);
    w.setClassInfo(arg);

    arg=readOption(narg, argc, 2, "-thu");
    w.setThu(arg);

    if(arg!=NULL)
    {
        arg=readOption(narg, argc, 2, "-srp");
        if(arg!=NULL) bsortres=atoi(arg);

        arg=readOption(narg, argc, 2, "-srt");
        if(arg!=NULL) 
        {
            switch(atoi(arg))
            {
            case -1:
                if(bsortres) w.m_grid->sortByResolution(false);
                else w.m_grid->sortByPercentage(false);
                break;
            case 0:
                break;
            case 1:
                if(bsortres) w.m_grid->sortByResolution(true);
                else w.m_grid->sortByPercentage(true);
                break;
            }
        }
    }

    //w.m_grid->setList("THUMTMP_SelectedPar.txt");
    w.m_grid->displayStack(0);


    w.show();

    return a.exec();
}
