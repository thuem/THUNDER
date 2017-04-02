/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#include "Symmetry.h"

using std::cout;
using std::endl;

Symmetry::Symmetry() {}

Symmetry::Symmetry(const char sym[])
{
    init(sym);
}

Symmetry::Symmetry(const int pgGroup,
                   const int pgOrder)
{
    _pgGroup = pgGroup;
    _pgOrder = pgOrder;

    init();
}

Symmetry::Symmetry(const Symmetry& that)
{
    *this = that;
}

Symmetry::~Symmetry()
{
    clear();
}

Symmetry& Symmetry::operator=(const Symmetry& that)
{
    clear();

    _pgGroup = that.pgGroup();
    _pgOrder = that.pgOrder();

    mat33 L, R;
    for (ptrdiff_t i = 0; i < that.nSymmetryElement(); i++)
    {
        that.get(L, R, i);
        append(L, R);
    }

    return *this;
}

void Symmetry::init(const char sym[])
{
    symmetryGroup(_pgGroup, _pgOrder, sym);

    init();
}

int Symmetry::pgGroup() const
{
    return _pgGroup;
}

int Symmetry::pgOrder() const
{
    return _pgOrder;
}

void Symmetry::get(mat33& L,
                   mat33& R,
                   const int i) const
{
    L = _L[i];
    R = _R[i];
}

int Symmetry::nSymmetryElement() const
{
    return _L.size();
}

int Symmetry::nFractionSpace() const
{
    switch (_pgGroup)
    {
        case PG_CN:
            return _pgOrder;
        
        case PG_CI:
        case PG_CS:
            return 2;

        case PG_CNV:
            return 4;

        case PG_CNH:
            return _pgOrder * 2;
        
        case PG_SN:
            return _pgOrder;

        case PG_DN:
            return _pgOrder * 2;

        case PG_DNV:
        case PG_DNH:
            return _pgOrder * 4;

        case PG_T:
            return 4;

        case PG_TD:
        case PG_TH:
            return 8;

        case PG_O:
            return 8;

        case PG_OH:
            return 16;

        case PG_I:
        case PG_I2:
        case PG_I1:
        case PG_I3:
        case PG_I4:
            return 20;

        case PG_IH:
        case PG_I2H:
        case PG_I1H:
        case PG_I3H:
        case PG_I4H:
            return 40;

        default:
            CLOG(FATAL, "LOGGER_SYS") << "UNKNOWN SYMMETRY";
            abort();
    }
}

void Symmetry::clear()
{
    _L.clear();
    _R.clear();
}

void Symmetry::init()
{
    vector<SymmetryOperation> entry;
    fillSymmetryEntry(entry, _pgGroup, _pgOrder);

    init(entry);
}

void Symmetry::init(const vector<SymmetryOperation>& entry)
{
    _L.clear();
    _R.clear();
    
    fillLR(entry);
    completePointGroup();
}

void Symmetry::append(const mat33& L,
                      const mat33& R)
{
    _L.push_back(L);
    _R.push_back(R);
}

void Symmetry::set(const mat33& L,
                   const mat33& R,
                   const int i)
{
    _L[i] = L;
    _R[i] = R;
}

void Symmetry::fillLR(const vector<SymmetryOperation>& entry)
{
    mat33 L, R;

    for (size_t i = 0; i < entry.size(); i++)
    {
        L.setIdentity();

        if (entry[i].id == 0)
        {
            // rotation

            double angle = 2 * M_PI / entry[i].fold;

            for (int j = 1; j < entry[i].fold; j++)
            {
                rotate3D(R, angle * j, entry[i].axisPlane);

                //if (novo(L, R)) append(L, R.transpose());
                if (novo(L, R)) append(L, R);
                //append(L, R);
            }
        }
        else if (entry[i].id == 1)
        {
            // reflexion

            reflect3D(R, entry[i].axisPlane);
            if (novo(L, R)) append(L, R);
        }
        else if (entry[i].id == 2)
        {
            // inversion
            
            /* L -> [ 1  0  0]
             *      [ 0  1  0]
             *      [ 0  0 -1]
             * R -> [-1  0  0]
             *      [ 0 -1  0]
             *      [ 0  0 -1] */

            L(2, 2) = -1;

            R = vec3(-1, -1, -1).asDiagonal();

            if (novo(L, R)) append(L, R);
        }
    }
}

bool Symmetry::novo(const mat33& L,
                    const mat33& R) const
{
    // check whether L and R are both identity matrix or not
    mat33 I = mat33::Identity();
    if (SAME_MATRIX(L, I) && SAME_MATRIX(R, I))
        return false;

    // check whether (L, R) exists in (_L, _R) or not
    for (size_t i = 0; i < _L.size(); i++)
        if (SAME_MATRIX(L, _L[i]) && SAME_MATRIX(R, _R[i]))
            return false;

    return true;
}

static bool completePointGroupHelper(umat& table,
                                     int& i,
                                     int& j)
{
    for (int row = 0; row < table.rows(); row++)
        for (int col = 0; col < table.cols(); col++)
            if (table(row, col) == 0)
            {
                i = row;
                j = col;
                table(row, col) = 1;

                return true;
            }

    return false;
}

void Symmetry::completePointGroup()
{
    umat table = umat::Zero(nSymmetryElement(),
                            nSymmetryElement());

    int i = -1, j = -1;
    while (completePointGroupHelper(table, i, j))
    {
        mat33 L = _L[i] * _L[j];
        mat33 R = _R[i] * _R[j];

        if (novo(L, R))
        {
            append(L, R);

            umat tableNew = umat::Zero(table.rows() + 1,
                                       table.cols() + 1);

            tableNew.topLeftCorner(table.rows(), table.cols()) = table;

            table = tableNew;
        }
    }
}

void display(const Symmetry& sym)
{
    mat33 L, R;

    for (int i = 0; i < sym.nSymmetryElement(); i++)
    {
        printf("%03d Symmetry Element:\n", i);

        sym.get(L, R, i);

        std::cout << "L matrix:\n" << L << std::endl << std::endl;
        std::cout << "R matrix:\n" << R << std::endl << std::endl;
    }
}

bool asymmetry(const Symmetry& sym)
{
    if ((sym.pgGroup() == PG_CN) &&
        (sym.pgOrder() == 1))
        return true;
    else
        return false;
}

bool asymmetryUnit(const vec3 dir,
                   const Symmetry& sym)
{
    double phi, theta;
    angle(phi, theta, dir);
    return asymmetryUnit(phi, theta, sym);
}

bool asymmetryUnit(const double phi,
                   const double theta,
                   const Symmetry& sym)
{
    return asymmetryUnit(phi, theta, sym.pgGroup(), sym.pgOrder());
}

bool asymmetryUnit(const double phi,
                   const double theta,
                   const int pgGroup,
                   const int pgOrder)
{
    // basic range check
    if ((phi < 0) ||
        (phi >= 2 * M_PI) ||
        (theta < 0) ||
        (theta > M_PI))
        return false;

    switch (pgGroup)
    {
        case PG_CN:
            return (phi <= 2 * M_PI / pgOrder);
        
        case PG_CI:
        case PG_CS:
            return (theta <= M_PI / 2);

        case PG_CNV:
            return (phi <= M_PI / pgOrder);

        case PG_CNH:
            return ((phi <= 2 * M_PI / pgOrder) &&
                    (theta <= M_PI / 2));
        
        case PG_SN:
            return ((phi <= 4 * M_PI / pgOrder) &&
                    (theta <= M_PI / 2));

        case PG_DN:
            return ((phi >= M_PI / 2) &&
                    (phi <= 2 * M_PI / pgOrder + M_PI / 2) &&
                    (theta <= M_PI / 2));

        case PG_DNV:
        case PG_DNH:
            return ((phi >= M_PI / 2) &&
                    (phi <= M_PI / pgOrder + M_PI / 2) &&
                    (theta <= M_PI / 2));

        case PG_T:
            return ASY_3(T, phi, theta);

        case PG_TD:
            return ASY_3(TD, phi, theta);

        case PG_TH:
            return ASY_3(TH, phi, theta);

        case PG_O:
            return ((phi >= M_PI / 4) &&
                    (phi <= 3 * M_PI / 4) &&
                    (theta <= M_PI / 2) &&
                    ASY_3(O, phi, theta));

        case PG_OH:
            return ((phi >= 3 * M_PI / 2) &&
                    (phi <= 7 * M_PI / 4) &&
                    (theta <= M_PI / 2) &&
                    ASY_3(O, phi, theta));

        case PG_I:
        case PG_I2:
            return ASY_3(I2, phi, theta);

        case PG_I1:
            return ASY_3(I1, phi, theta);

        case PG_I3:
            return ASY_3(I3, phi, theta);

        case PG_I4:
            return ASY_3(I4, phi, theta);

        case PG_IH:
        case PG_I2H:
            return ASY_4(I2H, phi, theta);

        case PG_I1H:
            return ASY_4(I1H, phi, theta);

        case PG_I3H:
            return ASY_4(I3H, phi, theta);

        case PG_I4H:
            return ASY_4(I4H, phi, theta);

        default:
            CLOG(FATAL, "LOGGER_SYS") << "UNKNOWN SYMMETRY";
            abort();
    }
    abort();
}

void symmetryCounterpart(double& phi,
                         double& theta,
                         const Symmetry& sym)
{
    vec3 dir;
    direction(dir, phi, theta);
    symmetryCounterpart(dir(0), dir(1), dir(2), sym);
    angle(phi, theta, dir);
}

void symmetryCounterpart(double& ex,
                         double& ey,
                         double& ez,
                         const Symmetry& sym)
{
    vec3 dir(ex, ey, ez);
    if (asymmetryUnit(dir, sym)) return;

    mat33 L, R;
    vec3 newDir;

    for (int i = 0; i < sym.nSymmetryElement(); i++)
    {
        sym.get(L, R, i);
        newDir = R * dir;

        if (asymmetryUnit(newDir, sym))
        {
            ex = newDir(0);
            ey = newDir(1);
            ez = newDir(2);

            return;
        }
    }

    CLOG(WARNING, "LOGGER_SYS") << "Unable to find SymmetryCounterpart";
}

void symmetryRotation(vector<mat33>& sr,
                      const mat33 rot,
                      const Symmetry* sym)
{
    sr.clear();

    sr.push_back(rot);

    if (sym == NULL) return;

    if (asymmetry(*sym)) return;

    mat33 L, R;

    for (int i = 0; i < sym->nSymmetryElement(); i++)
    {
        sym->get(L, R, i);
        
        sr.push_back(R * rot);
    }
}
