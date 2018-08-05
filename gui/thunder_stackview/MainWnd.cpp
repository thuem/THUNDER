#include "MainWnd.h"


MainWnd::MainWnd(QWidget *parent)
    : QWidget(parent)
{

	m_LayoutWnd = new QGridLayout;

	QVBoxLayout *gridLeft=new QVBoxLayout;
	int bwidth=100;
	m_prevpage=new QPushButton("Prev Page");
	//m_prevpage->setStyleSheet("border: 2px solid #800000;");
	m_prevpage->setFixedWidth(bwidth);
	m_prevpage->setFocusPolicy(Qt::NoFocus);

	m_nextpage=new QPushButton("Next Page");	
	//m_nextpage->setStyleSheet("border: 2px solid #800000;");
	m_nextpage->setFixedWidth(bwidth);
	m_nextpage->setFocusPolicy(Qt::NoFocus);

	m_zoom=new QPushButton("Zoom");	
	m_zoom->setFixedWidth(45);
	m_zoom->setFocusPolicy(Qt::NoFocus);
	m_bin=new QDoubleSpinBox;
	m_bin->setFixedWidth(bwidth/2);
	//m_bin->setFocusPolicy(Qt::NoFocus);
	m_bin->setMinimum(0.25);
	m_bin->setMaximum(4);
	m_bin->setSingleStep(0.25);
	m_bin->setValue(1.0);
	QGridLayout *hgrid_1=new QGridLayout;
	hgrid_1->addWidget(m_zoom,0,0,Qt::AlignLeft);
	hgrid_1->addWidget(m_bin,0,1,Qt::AlignLeft);
	hgrid_1->setSizeConstraint(QLayout::SetFixedSize);
	//hgrid_1->addStretch();

	m_contrasti=new QPushButton("Contrast +");
	m_contrasti->setFixedWidth(bwidth);
	m_contrasti->setFocusPolicy(Qt::NoFocus);

	m_contrastd=new QPushButton("Contrast -");	
	m_contrastd->setFixedWidth(bwidth);
	m_contrastd->setFocusPolicy(Qt::NoFocus);


	setStyleSheet(
         "QGroupBox"
         "{"
         "    border: 1px solid lightGray;"
         "    margin-top: 1.2ex;"
         "}"
                );
	m_groupSort=new QGroupBox("Sort By");
	m_groupSort->setFixedWidth(bwidth);
	m_groupSort->setFocusPolicy(Qt::NoFocus);
	m_comboSort=new QComboBox;	
	m_comboSort->addItem(QString("Resolution"));
	m_comboSort->addItem(QString("Percentage"));
	m_comboSort->setFixedWidth(80);
	m_comboSort->setFocusPolicy(Qt::NoFocus);
	m_ascend=new QPushButton("Ascending");
	m_ascend->setFixedWidth(80);
	m_ascend->setFocusPolicy(Qt::NoFocus);
	m_descend=new QPushButton("Descending");
	m_descend->setFixedWidth(80);
	m_descend->setFocusPolicy(Qt::NoFocus);
	QGridLayout *hgroup_1=new QGridLayout;
	hgroup_1->addWidget(m_comboSort,0,0);
	hgroup_1->addWidget(m_ascend,1,0);
	hgroup_1->addWidget(m_descend,2,0);
	m_groupSort->setLayout(hgroup_1);

	m_groupSave=new QGroupBox("Selected");
	m_groupSave->setFixedWidth(bwidth);
	m_groupSave->setFocusPolicy(Qt::NoFocus);
	m_savelist=new QPushButton("Save List");
	m_savelist->setFixedWidth(80);
	m_savelist->setFocusPolicy(Qt::NoFocus);
	m_savepar=new QPushButton("Save thu");
	m_savepar->setFixedWidth(80);
	m_savepar->setFocusPolicy(Qt::NoFocus);
	QGridLayout *hgroup_2=new QGridLayout;
	hgroup_2->addWidget(m_savelist,0,0);
	hgroup_2->addWidget(m_savepar,1,0);
	m_groupSave->setLayout(hgroup_2);


	gridLeft->addWidget(m_prevpage,0,Qt::AlignTop);	
	gridLeft->addWidget(m_nextpage,0,Qt::AlignTop);	
	gridLeft->addLayout(hgrid_1,0);
	gridLeft->addWidget(m_contrasti,0,Qt::AlignTop);	
	gridLeft->addWidget(m_contrastd,0,Qt::AlignTop);	
	gridLeft->addWidget(m_groupSort,0,Qt::AlignTop);
	gridLeft->addWidget(m_groupSave,0,Qt::AlignTop);
	gridLeft->addStretch();

	m_LayoutWnd->addLayout(gridLeft,0,0);
	m_LayoutWnd->setColumnStretch(0,0);
	m_LayoutWnd->setColumnStretch(1,1);

	m_grid = new THUImGrid;
	//image.readlist("MouseLeft.txt","MouseRight.txt");
    //if(argc==8) image.setBox(atoi(argv[6]),atoi(argv[7]),atoi(argv[5]));

    //if(argc==6) image.readOutputParFile(argv[5]);
    //if(/argc==9) image.readOutputParFile(argv[8]);
    m_grid->show(); 
	m_LayoutWnd->addWidget(m_grid,0,1);

	setLayout(m_LayoutWnd);

	setFocusPolicy(Qt::NoFocus);
	m_grid->setFocus();

	connect(m_grid, SIGNAL(OnTitleChanged(QString)),this,SLOT(OnChangeTitle(QString)));
	connect(m_prevpage,SIGNAL(clicked()),this,SLOT(OnPageUp()));
	connect(m_nextpage,SIGNAL(clicked()),this,SLOT(OnPageDown()));
	connect(m_zoom,SIGNAL(clicked()),this,SLOT(OnZoom()));
	connect(m_contrasti,SIGNAL(clicked()),this,SLOT(OnContrastInc()));
	connect(m_contrastd,SIGNAL(clicked()),this,SLOT(OnContrastDec()));
	connect(m_ascend,SIGNAL(clicked()),this,SLOT(OnSortAscend()));
	connect(m_descend,SIGNAL(clicked()),this,SLOT(OnSortDescend()));
	connect(m_savelist,SIGNAL(clicked()),this,SLOT(OnSaveList()));
	connect(m_savepar,SIGNAL(clicked()),this,SLOT(OnSavePar()));
}


MainWnd::~MainWnd()
{

}

void MainWnd::setZoom(float val)
{
	m_bin->setValue(val);
}

void MainWnd::OnChangeTitle(QString title)
{
	setWindowTitle(title);
}
void MainWnd::OnPageUp()
{
	int curpage=m_grid->getCurPage()-1;
	if(curpage<0) return;
    m_grid->displayStack(curpage);
}
void MainWnd::OnPageDown()
{
	int curpage=m_grid->getCurPage()+1;
    m_grid->displayStack(curpage);
}
void MainWnd::OnZoom()
{
	float bin=1.0/m_bin->value();
	m_grid->setBin(bin);
}
void MainWnd::OnContrastInc()
{
	float std=m_grid->getNStd()-0.25;
	if(std<=0.1) return;
	m_grid->setNStd(std);
}
void MainWnd::OnContrastDec()
{
	float std=m_grid->getNStd()+0.25;
	m_grid->setNStd(std);
}

void MainWnd::OnSortAscend()
{
	bool bascend=true;
	//printf("selected = %d\n",m_comboSort->currentIndex());
	switch(m_comboSort->currentIndex()) 
	{
	case 0:
		m_grid->sortByResolution(bascend);
		break;
    case 1: 
    	m_grid->sortByPercentage(bascend);
    	break;
    default:
    	return;
	}
	m_grid->displayStack(0);
}

void MainWnd::OnSortDescend()
{
	bool bascend=false;
	//printf("selected = %d\n",m_comboSort->currentIndex());
	switch(m_comboSort->currentIndex()) 
	{
	case 0:
		m_grid->sortByResolution(bascend);
		break;
    case 1: 
    	m_grid->sortByPercentage(bascend);
    	break;
    default:
    	return;
	}
	m_grid->displayStack(0);
}

void MainWnd::OnSaveList()
{
    QFileDialog dlg(NULL,"Save as","./ListSelected.txt","List File(*.txt);;All File(*.*)");
    dlg.setAcceptMode(QFileDialog::AcceptSave);
    dlg.setFileMode(QFileDialog::AnyFile);
	
    if(dlg.exec())
    {
        FILE *fp=fopen(dlg.selectedFiles()[0].toStdString().c_str(),"w");
        if(fp==NULL)
        {
        	printf("Warning: Failed to create file %s\n",dlg.selectedFiles()[0].toStdString().c_str());
        	return;
        }
        int i;
        vector<int> list=m_grid->getSelected();
        for(i=0;i<list.size();i++) fprintf(fp,"%d\n",list[i]);
        fclose(fp);
    	printf("Saved list to file %s\n",dlg.selectedFiles()[0].toStdString().c_str());
    }

}

void MainWnd::setClassInfo(char *fnClassInfo)
{
	if(fnClassInfo==NULL) return;
	m_fnClassInfo=fnClassInfo;
}

void MainWnd::setThu(char* thu)
{
	if(thu==NULL) return;
	m_fnThu=thu;
}

bool MainWnd::isInList(int classnum, vector<int> &list)
{
	int i;
	for(i=0;i<list.size();i++) if(classnum==list[i]) return true;
	return false;	
}

void MainWnd::OnSavePar()
{
	vector<int> list=m_grid->getSelected();
	if(list.size()==0) 
	{
		printf("Warning: No classes are selected. \n");
		return; 
	}

	FILE *fthu=fopen(m_fnThu.c_str(),"r");
	if(fthu==NULL) 
	{
		printf("Warning: No *.thu file are loaded. \n");
		return;
	}

	char line[4096], lineraw[4096];
	char* word;
	int i;
	int classnum;

	QFileDialog dlg(NULL,"Save as","./ParticleSelected.thu","thu File(*.thu);;All File(*.*)");
    dlg.setAcceptMode(QFileDialog::AcceptSave);
    dlg.setFileMode(QFileDialog::AnyFile);
	
    if(dlg.exec())
    {
        FILE *fp=fopen(dlg.selectedFiles()[0].toStdString().c_str(),"w");
        if(fp==NULL)
        {
        	printf("Warning: Failed to create file %s\n",dlg.selectedFiles()[0].toStdString().c_str());
        	fclose(fthu);
        	return;
        }
        int count=0;
        while(!feof(fthu))
        {
        	if(fgets(line,4096,fthu)==NULL) continue;
        	strcpy(lineraw, line);
    		word = strtok(line, " ");
    		for (i = 0; i < 12; i++) word = strtok(NULL, " ");
			classnum=atoi(word);
			if(isInList(classnum, list)) 
			{
				fputs(lineraw,fp);
				count++;
			}
        }


        fclose(fp);
    	printf("%d particles are saved to file %s\n",count,dlg.selectedFiles()[0].toStdString().c_str());
    }

    fclose(fthu);
}