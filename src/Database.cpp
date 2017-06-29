/*******************************************************************************
 * Author: Mingxu Hu, Bing Li, Siyuan Ren
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include "Database.h"

Database::Database()
{
    _db = NULL;
}

Database::Database(const char database[])
{
    openDatabase(database);
}

Database::~Database()
{
    fclose(_db);
}

void Database::openDatabase(const char database[])
{
    _db = fopen(database, "r");

    if (_db == NULL) REPORT_ERROR("FAIL TO OPEN DATABASE");
}

void Database::saveDatabase(const char database[])
{
    // TODO
}

int Database::nParticle() const
{
    rewind(_db);

    int result = 0;

    char line[FILE_LINE_LENGTH];

    IF_MASTER
    {
        while (fgets(line, FILE_LINE_LENGTH - 1, _db)) result++;
    }

    MPI_Bcast(&result, 1, MPI_INT, MASTER_ID, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    return result;
}

int Database::nGroup() const
{
    rewind(_db);

    int result = 0;

    char line[FILE_LINE_LENGTH];
    char* word;

    IF_MASTER
    {
        while (fgets(line, FILE_LINE_LENGTH - 1, _db))
        {
            word = strtok(line, " ");

            for (int i = 0; i < THU_GROUP_ID; i++)
                word = strtok(NULL, " ");

            if (atoi(word) > result)
                result = atoi(word);
        }
    }

    MPI_Bcast(&result, 1, MPI_INT, MASTER_ID, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    return result;
}

int Database::nParticleRank()
{
    IF_MASTER
    {
        CLOG(WARNING, "LOGGER_SYS") << "NO PARTICLE ASSIGNED TO MASTER PROCESS";

        return 0;
    }

    return (_end - _start + 1);
}

void Database::assign()
{
    split(_start, _end, _commRank);
}

void Database::index()
{
    _offset.resize(nParticle());

    rewind(_db);

    char line[FILE_LINE_LENGTH];

    IF_MASTER
    {
        for (int i = 0; i < (int)_offset.size(); i++)
        {
            _offset[i] = ftell(_db);

            fgets(line, FILE_LINE_LENGTH - 1, _db);
        }
    }

    MPI_Bcast(&_offset[0], _offset.size(), MPI_LONG, MASTER_ID, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
}

void Database::shuffle()
{
    _reg.resize(nParticle());

    IF_MASTER
    {
        for (int i = 0; i < (int)_reg.size(); i++)
            _reg[i] = i;

#ifdef DATABASE_SHUFFLE
        gsl_rng* engine = get_random_engine();

        gsl_ran_shuffle(engine, &_reg[0], _reg.size(), sizeof(int));
#endif
    }

    MPI_Bcast(&_reg[0], _reg.size(), MPI_INT, MASTER_ID, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
}

long Database::offset(const int i) const
{
    return _offset[_reg[i]];
}

double Database::coordX(const int i) const
{
    fseek(_db, _offset[_reg[i]], SEEK_SET);

    char line[FILE_LINE_LENGTH];
    char* word;

    fgets(line, FILE_LINE_LENGTH - 1, _db);

    word = strtok(line, " ");

    for (int i = 0; i < THU_COORDINATE_X; i++)
        word = strtok(NULL, " ");

    return atoi(word);
}

double Database::coordY(const int i) const
{
    fseek(_db, _offset[_reg[i]], SEEK_SET);

    char line[FILE_LINE_LENGTH];
    char* word;

    fgets(line, FILE_LINE_LENGTH - 1, _db);

    word = strtok(line, " ");

    for (int i = 0; i < THU_COORDINATE_Y; i++)
        word = strtok(NULL, " ");

    return atoi(word);
}

int Database::groupID(const int i) const
{
    fseek(_db, _offset[_reg[i]], SEEK_SET);

    char line[FILE_LINE_LENGTH];
    char* word;

    fgets(line, FILE_LINE_LENGTH - 1, _db);

    word = strtok(line, " ");

    for (int i = 0; i < THU_GROUP_ID; i++)
        word = strtok(NULL, " ");

    return atoi(word);
}

string Database::path(const int i) const
{
    fseek(_db, _offset[_reg[i]], SEEK_SET);

    char line[FILE_LINE_LENGTH];
    char* word;

    fgets(line, FILE_LINE_LENGTH - 1, _db);

    word = strtok(line, " ");

    for (int i = 0; i < THU_PARTICLE_PATH; i++)
        word = strtok(NULL, " ");

    return string(word);
}

string Database::micrographPath(const int i) const
{
    fseek(_db, _offset[_reg[i]], SEEK_SET);

    char line[FILE_LINE_LENGTH];
    char* word;

    fgets(line, FILE_LINE_LENGTH - 1, _db);

    word = strtok(line, " ");

    for (int i = 0; i < THU_MICROGRAPH_PATH; i++)
        word = strtok(NULL, " ");

    return string(word);
}

void Database::ctf(double& voltage,
                   double& defocusU,
                   double& defocusV,
                   double& defocusTheta,
                   double& Cs,
                   double& amplitudeConstrast,
                   double& phaseShift,
                   const int i) const
{
    fseek(_db, _offset[_reg[i]], SEEK_SET);

    char line[FILE_LINE_LENGTH];
    char* word;

    fgets(line, FILE_LINE_LENGTH - 1, _db);

    word = strtok(line, " ");

    voltage = atof(word);

    word = strtok(NULL, " ");

    defocusU = atof(word);

    word = strtok(NULL, " ");

    defocusV = atof(word);

    word = strtok(NULL, " ");

    defocusTheta = atof(word);

    word = strtok(NULL, " ");

    Cs = atof(word);

    word = strtok(NULL, " ");

    amplitudeConstrast = atof(word);

    word = strtok(NULL, " ");

    phaseShift = atof(word);
}

void Database::ctf(CTFAttr& dst,
                   const int i) const
{
    ctf(dst.voltage,
        dst.defocusU,
        dst.defocusV,
        dst.defocusTheta,
        dst.Cs,
        dst.amplitudeContrast,
        dst.phaseShift,
        i);
}

int Database::cls(const int i) const
{
    fseek(_db, _offset[_reg[i]], SEEK_SET);

    char line[FILE_LINE_LENGTH];
    char* word;

    fgets(line, FILE_LINE_LENGTH - 1, _db);

    word = strtok(line, " ");

    for (int i = 0; i < THU_CLASS_ID; i++)
        word = strtok(NULL, " ");

    return atoi(word);
}

vec4 Database::quat(const int i) const
{
    fseek(_db, _offset[_reg[i]], SEEK_SET);

    char line[FILE_LINE_LENGTH];
    char* word;

    fgets(line, FILE_LINE_LENGTH - 1, _db);

    word = strtok(line, " ");

    for (int i = 0; i < THU_QUATERNION_0; i++)
        word = strtok(NULL, " ");

    vec4 result;

    result(0) = atof(word);

    word = strtok(NULL, " ");

    result(1) = atof(word);

    word = strtok(NULL, " ");

    result(2) = atof(word);

    word = strtok(NULL, " ");

    result(3) = atof(word);

    return result;
}

double Database::stdR(const int i) const
{
    fseek(_db, _offset[_reg[i]], SEEK_SET);

    char line[FILE_LINE_LENGTH];
    char* word;

    fgets(line, FILE_LINE_LENGTH - 1, _db);

    word = strtok(line, " ");

    for (int i = 0; i < THU_STD_ROTATION; i++)
        word = strtok(NULL, " ");

    return atof(word);
}

vec2 Database::tran(const int i) const
{
    fseek(_db, _offset[_reg[i]], SEEK_SET);

    char line[FILE_LINE_LENGTH];
    char* word;

    fgets(line, FILE_LINE_LENGTH - 1, _db);

    word = strtok(line, " ");

    for (int i = 0; i < THU_TRANSLATION_X; i++)
        word = strtok(NULL, " ");

    vec2 result;

    result(0) = atof(word);

    word = strtok(NULL, " ");

    result(1) = atof(word);

    return result;
}

double Database::stdTX(const int i) const
{
    fseek(_db, _offset[_reg[i]], SEEK_SET);

    char line[FILE_LINE_LENGTH];
    char* word;

    fgets(line, FILE_LINE_LENGTH - 1, _db);

    word = strtok(line, " ");

    for (int i = 0; i < THU_STD_TRANSLATION_X; i++)
        word = strtok(NULL, " ");

    return atof(word);
}

double Database::stdTY(const int i) const
{
    fseek(_db, _offset[_reg[i]], SEEK_SET);

    char line[FILE_LINE_LENGTH];
    char* word;

    fgets(line, FILE_LINE_LENGTH - 1, _db);

    word = strtok(line, " ");

    for (int i = 0; i < THU_STD_TRANSLATION_Y; i++)
        word = strtok(NULL, " ");

    return atof(word);
}

double Database::d(const int i) const
{
    fseek(_db, _offset[_reg[i]], SEEK_SET);

    char line[FILE_LINE_LENGTH];
    char* word;

    fgets(line, FILE_LINE_LENGTH - 1, _db);

    word = strtok(line, " ");

    for (int i = 0; i < THU_DEFOCUS_FACTOR; i++)
        word = strtok(NULL, " ");

    return atof(word);
}

double Database::stdD(const int i) const
{
    fseek(_db, _offset[_reg[i]], SEEK_SET);

    char line[FILE_LINE_LENGTH];
    char* word;

    fgets(line, FILE_LINE_LENGTH - 1, _db);

    word = strtok(line, " ");

    for (int i = 0; i < THU_STD_DEFOCUS_FACTOR; i++)
        word = strtok(NULL, " ");

    return atof(word);
}

void Database::split(int& start,
                     int& end,
                     const int commRank)
{
    int size = nParticle();

    IF_MASTER return;

    int piece = size / (_commSize - 1);

    if (commRank <= size % (_commSize - 1))
    {
        start = (piece + 1) * (commRank - 1);
        end = start + (piece + 1) - 1;
    }
    else
    {
        start = piece * (commRank - 1) + size % (_commSize - 1);
        end = start + piece - 1;
    }
}
