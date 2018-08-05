#pragma once
#include <string>
#include <vector>
#include <algorithm>
#include <string.h>
#include <math.h>

using namespace std;


bool isComment(const char* str, char symbol='#');


char* readOption(int narg, char *argc[], int startnum, const char *option);

// im must be free by the user.  axis=[0,1,2]->[x,y,z] , slice is cound form the center.
bool readMRCSection(float **im, int &dimx, int &dimy, const char* filename, int axis, int slice=0);

void float2uchar(unsigned char *dst, float *src, int width, int height, float nstd);

//operate string
string RmExtName(string fullname);
string PickFileName(string fullname);
string PickMainFileName(string fullname);
string trimString(string str);
string trimString(string str, char c);
string cropString(string str, char c);

// phrase string like:   name = value
// if fail to read both name and value, return false.
bool getKeyAndValue(string str, string &key, string &value, string pattern="=");

//the format of input g is like: 1,2,3  or  1,2:7,8  or 1,2:2:10,11  (0-based, no space between numbers)
vector<string> split(string str,string pattern);
void getNumList(const char* g, vector<int> &list);

//phrase 1,2:4,5@filename to NumList and filename
bool getNumAndFile(const char* g, vector<int> &list, string &fname);

//For THUNDER ClassInfo file
struct ClassInfo
{
        int id;
        float resolution;
        float percentage;
};
bool compareId(ClassInfo c1, ClassInfo c2);
bool compareResolution(ClassInfo c1, ClassInfo c2);
bool comparePercentage(ClassInfo c1, ClassInfo c2);

//For THUNDER ClassInfo file
class RoundInfo
{
public:
	RoundInfo(){id=0;};
	RoundInfo(const RoundInfo &r);
	RoundInfo &operator=(const RoundInfo &r);
	bool readClassInfo(const char* classinfofile);
public:	
	int id;
	vector<ClassInfo> info;

public:
	void sortById(bool bascending=true);
	void sortByResolution(bool bascending=true);
	void sortByPercentage(bool bascending=false);
};

//For THUNDER Default filename
enum THUNDER_FileType
{
	THU_ClassInfo,
	THU_FSC,
	THU_Meta,
	THU_Reference,
	THU_Reference_Class3D
};
vector<string> THUNDER_GetFileName(THUNDER_FileType type, int roundId, int classId=0);
