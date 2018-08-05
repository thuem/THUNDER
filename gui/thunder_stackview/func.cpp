#include "func.h"
#include "mrc.h"

bool isComment(const char* str, char symbol)
{
	int i, len;
    len = strlen(str) ;
    for(i=0;i<len;i++)
    {
        if( str[i] != ' ') break ;
    }
    if(str[i]==symbol) return true;
    else return false;
}

char* readOption(int narg, char *argc[], int startnum, const char *option)
{
	string arg;
	int i;
    for(i=startnum;i<narg;i+=2)
    {
        arg=argc[i];
        if(arg.compare(option)==0)
        {
            i++;
            if(i>=narg) return NULL;
            else return argc[i];
        }
    }
    return NULL;
}

// im must be free by the user.  axis=[0,1,2]->[x,y,z] , slice is cound form the center.
bool readMRCSection(float **im, int &dimx, int &dimy, const char* filename, int axis, int slice)
{
	MRC mrc;
	if(mrc.open(filename, "rb")<=0)
	{
		printf("Warning: Failed to read %s\n",filename);
		return false;
	}
	int nx=mrc.getNx();
	int ny=mrc.getNy();
	int nz=mrc.getNz();

	float *buf=NULL;
	int ns,i,j;
	switch(axis)
	{
	case 0: //along x
		dimx=ny;
		dimy=nz;
		ns=slice+nx/2;
		if(ns<0) ns=0;
		if(ns>=nx) ns=nx-1;
		buf=new float[dimx*dimy];
		for(j=0;j<dimy;j++)
			for(i=0;i<dimx;i++)
			{
				mrc.readPixel(&buf[i+j*dimx], j, i, ns);
			}
		break;

	case 1: //along y
		dimx=nx;
		dimy=nz;
		ns=slice+ny/2;
		if(ns<0) ns=0;
		if(ns>=nx) ns=nx-1;
		buf=new float[dimx*dimy];
		for(j=0;j<dimy;j++)
			for(i=0;i<dimx;i++)
			{
				mrc.readPixel(&buf[i+j*dimx], j, ns, i);
			}
		break;

	case 2: //along z
		dimx=nx;
		dimy=ny;
		ns=slice+nz/2;
		if(ns<0) ns=0;
		if(ns>=nx) ns=nx-1;
		buf=new float[dimx*dimy];
		mrc.read2DIm_32bit(buf,ns);
		break;
	}
	mrc.close();

	*im=buf;

	return true;
}

void float2uchar(unsigned char *dst, float *src, int width, int height, float nstd)
{
        int size=width*height;
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
        float min=mean-std*nstd;
        float max=mean+std*nstd;
        float scale=255/(max-min);
        for(i=0;i<size;i++)
        {
                if(src[i]<=min) dst[i]=0;
                else if(src[i]>=max) dst[i]=255;
                else dst[i]=(unsigned char)((src[i]-min)*scale+0.5);
        }
}

//operate string
string RmExtName(string fullname)
{
        size_t pos = fullname.find_last_of(".");
        return fullname.substr(0,pos);
}

string PickFileName(string fullname)
{
        size_t pos = fullname.find_last_of("/");
        return fullname.substr(pos+1);
}

string PickMainFileName(string fullname)
{
        return PickFileName(RmExtName(fullname));
}

string trimString(string str)
{
	string s = trimString(str,'\n');
    return trimString(s,' ');
}

string trimString(string str, char c)
{
	if(str.length()==0) return str;
    size_t s = str.find_first_not_of(c);
    size_t e = str.find_last_not_of(c);
    if(s==string::npos || e==string::npos) return "";
    return str.substr(s,e-s+1);
}

string cropString(string str, char c)
{
	if(str.length()<=2) return str;
    size_t s = str.find_first_of(c);
    size_t e = str.find_last_of(c);
    if(s==string::npos || e==string::npos) return str;
    if(e-s<=1) return "";
    return str.substr(s+1,e-s-1);
}

// phrase string like:   name = value  or "name" = "value"
// if fail to read both name and value, return false.
bool getKeyAndValue(string str, string &key, string &value, string pattern)
{
	vector<string> v=split(str,pattern);
    if(v.size()!=2) return false;
    key=trimString(cropString(v[0],'"'));
    value=trimString(cropString(v[1],'"'));
    return true;
}

vector<string> split(string str,string pattern) 
{   
	string::size_type pos;   
	vector<string> result;   
	str+=pattern;  
	int size=str.size();     
	for(int i=0; i<size; i++)   
	{     
		pos=str.find(pattern,i);     
		if(pos<size)     
		{       
			string s=str.substr(i,pos-i);       
			result.push_back(s);       
			i=pos+pattern.size()-1;     
		}   
	}   
	return result; 
} 

void getNumList(const char* g, vector<int> &list)
{
	list.clear();
	
	string str = g;
	vector<string> gnum = split(str,",");
	vector<string> gnumr;

	int n,n1,n2,step,i,j;
	for(i=0;i<gnum.size();i++)
	{
		gnumr= split(gnum[i],":");
		if(gnumr.size()==1)
		{
			n=atoi(gnumr[0].c_str());
			list.push_back(n);
		}
		else if(gnumr.size()==2)
		{
			n1=atoi(gnumr[0].c_str());
			n2=atoi(gnumr[1].c_str());
			
			if(n1<=n2) for(j=n1;j<=n2;j++) list.push_back(j);
			else for(j=n2;j<=n1;j++) list.push_back(j);
		}
		else if(gnumr.size()==3)
		{
			n1=atoi(gnumr[0].c_str());
			step=atoi(gnumr[1].c_str());
			n2=atoi(gnumr[2].c_str());
			
			if(n1<=n2 && step>0) for(j=n1;j<=n2;j+=step) list.push_back(j);
			else if(n1>=n2 && step<0) for(j=n2;j<=n1;j-=step) list.push_back(j);
		}
	}

}

//phrase 1,2:4,5@filename to NumList and filename
bool getNumAndFile(const char* g, vector<int> &list, string &fname)
{
	string str = g;
	vector<string> fp = split(str,"@");

	list.clear();
	fname="";
	if(fp.size()==1)
	{
		fname=fp[0];
		return true;
	}
	else if(fp.size()==2)
	{
		getNumList(fp[0].c_str(), list);
		fname=fp[1];
		return true;
	}
	return false;
}


//For THUNDER ClassInfo
bool compareId(ClassInfo c1, ClassInfo c2)
        {
                return c1.id<c2.id;
        }
bool compareResolution(ClassInfo c1, ClassInfo c2)
        {
                if(c1.resolution==c2.resolution) return c1.id<c2.id;
                else return c1.resolution<c2.resolution;
        }
bool comparePercentage(ClassInfo c1, ClassInfo c2)
        {
                if(c1.percentage==c2.percentage) return c1.id<c2.id;
                return c1.percentage<c2.percentage;
        }


//For THUNDER Round Info
RoundInfo::RoundInfo(const RoundInfo &r)
{
	this->id = r.id;
	this->info = r.info;
}

RoundInfo &RoundInfo::operator=(const RoundInfo &r)
{
	if ( this == &r ) return *this;
	this->id = r.id;
	this->info = r.info;
	return *this;
}

bool RoundInfo::readClassInfo(const char* classinfofile)
{
        if(classinfofile==NULL) return false;
        FILE *fp=fopen(classinfofile,"r");
        if(fp==NULL)
        {
                printf("Warnning: Failed to read %s\n",classinfofile);
                return false;
        }

        char line[256];
        ClassInfo ci;
       	info.clear();
        while(!feof(fp))
        {
                if(!fgets(line,256,fp)) continue;
                if(sscanf(line,"%d %f %f",&ci.id, &ci.resolution,&ci.percentage)!=3) continue;
                info.push_back(ci);
        }
        fclose(fp);

        return true;
}

void RoundInfo::sortById(bool bascending)
{
        if(info.size()==0) return;
        if(bascending) sort(info.begin(),info.end(),compareId);
        else sort(info.rbegin(),info.rend(),compareId);
}

void RoundInfo::sortByResolution(bool bascending)
{
        if(info.size()==0) return;
        if(bascending) sort(info.begin(),info.end(),compareResolution);
        else sort(info.rbegin(),info.rend(),compareResolution);
}

void RoundInfo::sortByPercentage(bool bascending)
{
        if(info.size()==0) return;
        if(bascending) sort(info.begin(),info.end(),comparePercentage);
        else sort(info.rbegin(),info.rend(),comparePercentage);
}



//For THUNDER Default filename
vector<string> THUNDER_GetFileName(THUNDER_FileType type, int roundId, int classId)
{
	char num[16];
	if(roundId < 0) sprintf(num,"Final");
	else sprintf(num,"Round_%03d",roundId);

	char cnum[16];
	sprintf(cnum,"Reference_%03d",classId);

	string pre,sub;
	vector<string> list;

	switch(type)
	{
	case THU_ClassInfo:
		pre="Class_Info_";
		sub+=".txt";
		list.push_back(pre+num+sub);
		return list;
	case THU_FSC:
		pre="FSC_";
		sub+=".txt";
		list.push_back(pre+num+sub);
		return list;
	case THU_Meta:
		pre="Meta_";
		sub+=".thu";
		list.push_back(pre+num+sub);
		return list;
	case THU_Reference:
		pre=cnum;
		sub+=".mrc";
		list.push_back(pre+"_A_"+num+sub);
		list.push_back(pre+"_B_"+num+sub);
		return list;
	case THU_Reference_Class3D:
		pre=cnum;
		sub+=".mrc";
		list.push_back(pre+"_"+num+sub);
		return list;
	}
	return list;
}