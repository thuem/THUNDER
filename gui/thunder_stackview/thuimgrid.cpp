#include "thuimgrid.h"
#include <QPainter>
#include <QMouseEvent>
#include <QRect>
#include "mrc.h"
#include <stdio.h>
#include <iostream>
#include "math.h"

THUImGrid::THUImGrid(QWidget *parent, int width, int height, float binning)
        : QWidget(parent)
{
	m_data=NULL;
	m_pwidth=0;
	m_pheight=0;
	m_ncol=0;
    m_nrow=0;
	init(width, height, binning);
    //initilize color table
    m_nstd=3;
    CreateColorTable();
 
}

THUImGrid::~THUImGrid()
{
    delete [] m_data;
}

bool THUImGrid::init(int width, int height, float binning)
{
	 m_winwidth=width;
	 m_winheight=height;
	 m_binning=binning;
	 //setFixedSize(m_winwidth,m_winheight);
	 
	 
    m_width=width*m_binning;
    m_height=height*m_binning;

    if(m_pwidth!=0 && m_pheight!=0)
    {
		m_ncol=m_width/m_pwidth;
    	m_nrow=m_height/m_pheight;
    }
 
 	if(m_data!=NULL) delete [] m_data;
    m_data=new uchar[m_width*m_height];
    m_curpage=0;
}

bool THUImGrid::setStack(const char *mrcstack, char* classinfo)
{
    if(m_stack.open(mrcstack,"rb")==0) return false;
    m_fnstack=mrcstack;

    m_pwidth=m_stack.getNx();
    m_pheight=m_stack.getNy();
    m_ncol=m_width/m_pwidth;
    m_nrow=m_height/m_pheight;
    
    m_xbox=m_stack.getNx();
	m_ybox=m_stack.getNy();
	m_boxstyle=0;

	if(m_info.size()==m_stack.getNz()) return true;
	ClassInfo ci;
	ci.resolution=0;
	ci.percentage=0;
	int i;
	m_info.clear();
	for(i=0;i<m_stack.getNz();i++)
	{
		ci.id=i;
		m_info.push_back(ci);
	}
	readClassInfo(classinfo);

    return true;
}

bool THUImGrid::readClassInfo(char* classinfo)
{
	if(classinfo==NULL) return false;
	FILE *fp=fopen(classinfo,"r");
	if(fp==NULL)
	{
		printf("Warnning: Failed to read %s\n",classinfo);
		return false;
	}

	char line[256];
	int id;
	float r,p;
	int count=0;
	while(!feof(fp))
	{
		if(!fgets(line,256,fp)) continue;
		if(sscanf(line,"%d %f %f",&id, &r,&p)!=3) continue;
		if(id<0 || id>=m_info.size())
		{
			printf("Warnning: The number of classes doesn't match the number of images.\n");
			continue;
		}
		m_info[id].resolution=r;
		m_info[id].percentage=p;
		count++;
	}
	fclose(fp);
	if(count!=m_info.size())
	{
		printf("Warnning: The number of classes is less than the number of images.\n");
	}

	
	return true;
}

void THUImGrid::sortByResolution(bool bascending)
{
	if(m_info.size()==0) return;
	if(bascending) sort(m_info.begin(),m_info.end(),compareResolution);
	else sort(m_info.rbegin(),m_info.rend(),compareResolution);
	//for(int i=0;i<m_info.size();i++) printf("%5d %12.6f %12.6f\n",
	//	m_info[i].id,m_info[i].resolution, m_info[i].percentage);
	//displayStack(0);resolution, m_info[i].per
}

void THUImGrid::sortByPercentage(bool bascending)
{
	if(m_info.size()==0) return;
	if(bascending) sort(m_info.begin(),m_info.end(),comparePercentage);
	else sort(m_info.rbegin(),m_info.rend(),comparePercentage);
	//for(int i=0;i<m_info.size();i++) printf("%5d %12.6f %12.6f\n",
	//	m_info[i].id,m_info[i].resolution, m_info[i].percentage);
	//displayStack(0);
}

void THUImGrid::setList(char *fn_selected)
{
	m_fselected=fn_selected;
	FILE *fp=fopen(fn_selected,"r");
	int x;
	if(fp!=NULL)
	{
		m_list.clear();
		while(fscanf(fp,"%d",&x)!=EOF)
			m_list.push_back(x-1);
	}
	if(fp!=NULL) fclose(fp);
}


void THUImGrid::setBox(int boxwidth, int boxheight, int boxstyle)
{
	m_xbox=boxwidth;
	m_ybox=boxheight;
	m_boxstyle=boxstyle;
	
}

void THUImGrid::paintEvent(QPaintEvent *event)
{
    QImage image(m_data,m_width,m_height,m_width,QImage::Format_Indexed8);
    image.setColorTable(m_colortable);
    //binning the display  
    QSize size(m_winwidth, m_winheight);
    QImage bimage=image.scaled(size,Qt::KeepAspectRatio,Qt::SmoothTransformation);
    
    QPainter painter(this);
    QPen pen(Qt::SolidLine);
    pen.setWidthF(1.5);
    painter.setPen(pen);
     

    painter.drawImage(0,0,bimage);

    int x,y,i,j,n,id;
    int start,end;
    start=m_curpage*m_ncol*m_nrow;
    end=(m_curpage+1)*m_ncol*m_nrow-1;
    if(end>=m_info.size()) end=m_info.size()-1;

    pen.setColor(QColor(0,200,0,255));
    painter.setPen(pen);
    for(i=start;i<=end;i++)
    {
    	id=m_info[i].id;
    	for(j=0;j<m_list.size();j++)
    	{
    		if(m_list[j]==id) break;
    	}
    	if(j==m_list.size()) continue;
    	n=i-start;
		x=n%m_ncol;
		y=n/m_ncol;
		x*=m_pwidth/m_binning;
		y*=m_pheight/m_binning;
		drawrect(&painter,x,y);
    }
/*    for(i=0;i<m_list.size();i++)
    {

    	  if(m_list[i]>=start && m_list[i]<=end)
    	  {
    	  	 n=m_list[i]-start;
		     x=n%m_ncol;
		     y=n/m_ncol;
		     x*=m_pwidth/m_binning;
		     y*=m_pheight/m_binning;
		     drawrect(&painter,x,y);
        }
    }
 */   

    // char text[32];
    // painter.setPen(QColor(255,255,0,255));
    // if(m_astigmatism.size()>0)
    // {
    //     for(i=start;i<=end;i++)
    //     {
    //         sprintf(text,"D%.0f R%.1f",m_astigmatism[i],m_PhaseRes[i]);
    //         n=i-start;
    //         x=n%m_ncol;
    //         y=n/m_ncol+1;
    //         x*=m_pwidth/m_binning;
    //         y*=m_pheight/m_binning;
    //         painter.drawText(x+1,y-1,QString(text));
    //     }
    // }

    painter.end();
}

void THUImGrid::mousePressEvent (QMouseEvent * event)
{
    QPoint curpos;
    int x,y,pos,i,j,id;
    curpos=event->pos();
    x=curpos.x()*m_binning/m_pwidth;
    y=curpos.y()*m_binning/m_pheight;
    if(x>=m_ncol) x=m_ncol-1;
    if(y>=m_nrow) y=m_nrow-1;
    pos=x+y*m_ncol+m_curpage*m_nrow*m_ncol;
    
    if(pos>=m_stack.getNz()) return;
    
    if(event->button() ==Qt::LeftButton)
    {
    	id=m_info[pos].id;
        for(i=0;i<m_list.size();i++)
        {
            if(m_list[i]==id) m_list.erase(m_list.begin()+i);
        }
        if(i==m_list.size()) m_list.push_back(id);
    }
    
    if(event->button() ==Qt::RightButton)
    {

    }

    repaint();

}

void THUImGrid::keyPressEvent ( QKeyEvent * event )
{
    string fplt;
    FILE *fp;
    int i;
    switch( event->key() )
    {

        case Qt::Key_PageUp:
            if(m_curpage<=0) return;
            m_curpage--;
            displayStack(m_curpage);
            break;

        case Qt::Key_Space:
        case Qt::Key_PageDown:
            if(m_curpage>=(m_stack.getNz()/(m_nrow*m_ncol)+1)) return;
            m_curpage++;
            displayStack(m_curpage);
            break;
        case Qt::Key_Return:

			      fp=fopen(m_fselected.c_str(),"w");
			      sort(m_list.begin(),m_list.end());
			      for(i=0;i<m_list.size();i++)
			      	fprintf(fp,"%d\n",m_list[i]+1);
			      fclose(fp);

            break;
         case Qt::Key_Left:
                m_curpage=m_curpage-20;
         	if(m_curpage<0) m_curpage=0;          
            displayStack(m_curpage);
         	break;
         	
         case Qt::Key_Right:
                m_curpage=m_curpage+20;
         	if(m_curpage>=(m_stack.getNz()/(m_nrow*m_ncol)-1)) 
         		m_curpage=(m_stack.getNz()/(m_nrow*m_ncol)-1);          
            displayStack(m_curpage);
         	break;

        //for modify contrast
        case Qt::Key_Plus:
            if(m_nstd<=0.5) break;
            m_nstd-=0.25;
            displayStack(m_curpage);
            break;
        case Qt::Key_Minus:
            if(m_nstd>=8) break;
            m_nstd+=0.25;
            displayStack(m_curpage);
            break;
    }


}

void THUImGrid::resizeEvent(QResizeEvent *event)
{
	QSize size = event->size();
	if(size==event->oldSize()) return;

	init(size.width(), size.height(), m_binning);
	displayStack(0);
}

void THUImGrid::drawrect(QPainter *painter, int x,int y)
{
	 int xshift=(m_pwidth-m_xbox)/m_binning/2;
	 int yshift=(m_pheight-m_ybox)/m_binning/2;
    QRect rect(x+1+xshift,y+1+yshift,m_xbox/m_binning-2,m_ybox/m_binning-2);
    if(m_boxstyle==0) painter->drawRect(rect);
    if(m_boxstyle==1) painter->drawEllipse(rect);
}

void THUImGrid::float2uchar(uchar *dst, float *src)
{
	int size=m_pwidth*m_pheight;
	if(size<=0) return;

	int i;
	float mean=0.0;
	float std=0.0,v;
	for(i=0;i<size;i++) mean+=src[i];
	mean/=size;
	for(i=0;i<size;i++) 
	{
		v=src[i]-mean;
		std+=v*v;
	}
	std=sqrt(std/size);
	float min=mean-std*m_nstd;
	float max=mean+std*m_nstd;
	float scale=255/(max-min);
	for(i=0;i<size;i++) 
	{
		if(src[i]<=min) dst[i]=0;
		else if(src[i]>=max) dst[i]=255;
		else dst[i]=uchar((src[i]-min)*scale+0.5);
	}
}


void THUImGrid::readMRC(int pagenum)
{
     memset(m_data,0,m_width*m_height);

    int i,xx,yy,Ix,Iy,Ibottom,Ileft,Ipos,Ppos;
    
    float *particlef=new float[m_pwidth*m_pheight];
    uchar *particle=new uchar[m_pwidth*m_pheight];  

    //x xx along vertical
    //y yy along horizen

    int particlenum;
    for(i=0;i<m_ncol*m_nrow;i++)
    {
        Ibottom=i/m_ncol*m_pheight;
        Ileft=i%m_ncol*m_pwidth;
        particlenum=pagenum*m_ncol*m_nrow+i;
        if(particlenum>=m_stack.getNz()) continue;
        m_stack.read2DIm(particlef,m_info[particlenum].id);
        float2uchar(particle,particlef);

        for(xx=0;xx<m_pheight;xx++)
            for(yy=0;yy<m_pwidth;yy++)
            {
                Ppos=xx*m_pwidth+yy;
                Ix=Ibottom+xx;
                Iy=Ileft+yy;
                Ipos=Ix*m_width+Iy;
                if(xx==0 || yy==0) 
                {
                	m_data[Ipos]=0;
                	continue;
                }
                m_data[Ipos]=particle[Ppos];
            }
    }
    
    delete [] particlef;
    delete [] particle;
}

void THUImGrid::displayStack(int pagenum)
{
	if(!m_stack.hasFile()) return;
	if(pagenum<0 || pagenum>m_stack.getNz()/(m_nrow*m_ncol)) return;
	m_curpage=pagenum;
	 
    readMRC(pagenum);

    char title[100];
    sprintf(title,"Stack: %s Page:%d/%d  Particle:%d-%d/%d  Contrast: %f",
          m_fnstack.c_str(),pagenum+1,m_stack.getNz()/(m_nrow*m_ncol)+1,
          pagenum*m_nrow*m_ncol+1,(pagenum+1)*m_nrow*m_ncol,m_stack.getNz(),m_nstd);
    QString name=title;
    emit OnTitleChanged(name);
    //setWindowTitle(name);

    repaint();
}

void THUImGrid::updateTitle()
{

    char minmax[100];
    sprintf(minmax,"Stack: %s  Page:%d/%d  Particle:%d-%d/%d   Contrast: %f",
          m_fnstack.c_str(),m_curpage+1,m_stack.getNz()/(m_nrow*m_ncol)+1,
          m_curpage*m_nrow*m_ncol+1,(m_curpage+1)*m_nrow*m_ncol,m_stack.getNz(),m_nstd);

    emit OnTitleChanged(QString(minmax));
    //setWindowTitle(QString(minmax));

}

void THUImGrid::CreateColorTable()
{
    m_colortable.clear();
    for(int i=0;i<256;i++)
    {
        m_colortable.push_back(qRgb(i,i,i));
    }
}

float THUImGrid::getNStd()
{
	return m_nstd;
}

int THUImGrid::getCurPage()
{
	return m_curpage;
}
float THUImGrid::getBin()
{
	return m_binning;
}

vector<int> THUImGrid::getSelected()
{
	vector<int> list;
	int i;
	for(i=0;i<m_list.size();i++) list.push_back(m_list[i]);
	sort(list.begin(),list.end());
	return list;
}

void THUImGrid::setNStd(float std)
{
	if(fabs(m_nstd-std)<=0.00001) return;
	m_nstd=std;
	displayStack(getCurPage());
}
void THUImGrid::setBin(float bin)
{
	init(m_winwidth,m_winheight,bin);
    setStack(m_fnstack.c_str());
    displayStack(0);
}

