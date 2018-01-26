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

    dmat33 L, R;
    for (ptrdiff_t i = 0; i < that.nSymmetryElement(); i++)
    {
        that.get(L, R, i);
        append(L, R);

        append(that.quat(i));
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

void Symmetry::get(dmat33& L,
                   dmat33& R,
                   const int i) const
{
    L = _L[i];
    R = _R[i];
}

dvec4 Symmetry::quat(const int i) const
{
    return _quat[i];
}

int Symmetry::nSymmetryElement() const
{
    return _L.size();
}

void Symmetry::clear()
{
    _L.clear();
    _R.clear();

    _quat.clear();
}

void Symmetry::init()
{
    vector<SymmetryOperation> entry;
    fillSymmetryEntry(entry, _pgGroup, _pgOrder);

    init(entry);
}

void Symmetry::init(const vector<SymmetryOperation>& entry)
{
    clear();
    
    fillLR(entry);
    completePointGroup();
}

void Symmetry::append(const dmat33& L,
                      const dmat33& R)
{
    _L.push_back(L);
    _R.push_back(R);
}

void Symmetry::append(const dvec4& quat)
{
    _quat.push_back(quat);
}

void Symmetry::set(const dmat33& L,
                   const dmat33& R,
                   const int i)
{
    _L[i] = L;
    _R[i] = R;
}

void Symmetry::set(const dvec4& quat,
                   const int i)
{
    _quat[i] = quat;
}

void Symmetry::fillLR(const vector<SymmetryOperation>& entry)
{
    dmat33 L, R;

    for (size_t i = 0; i < entry.size(); i++)
    {
        L.setIdentity();

        if (entry[i].id == 0)
        {
            // rotation

            RFLOAT angle = 2 * M_PI / entry[i].fold;

            for (int j = 1; j < entry[i].fold; j++)
            {
                rotate3D(R, angle * j, entry[i].axisPlane);

                if (novo(L, R))
                {
                    append(L, R);

                    dvec4 quat;
                    quaternion(quat, R);
                    append(quat);
                }

                /***
                if (novo(L, R.transpose()))
                {
                    append(L, R.transpose());

                    dvec4 quat;
                    quaternion(quat, R.transpose());
                    append(quat);
                }
                ***/
            }
        }
        else if (entry[i].id == 1)
        {
            CLOG(FATAL, "LOGGER_SYS") << "WRONG!";

            // reflexion

            reflect3D(R, entry[i].axisPlane);
            if (novo(L, R)) append(L, R);
        }
        else if (entry[i].id == 2)
        {
            CLOG(FATAL, "LOGGER_SYS") << "WRONG!";

            // inversion
            
            /* L -> [ 1  0  0]
             *      [ 0  1  0]
             *      [ 0  0 -1]
             * R -> [-1  0  0]
             *      [ 0 -1  0]
             *      [ 0  0 -1] */

            L(2, 2) = -1;

            R = dvec3(-1, -1, -1).asDiagonal();

            if (novo(L, R)) append(L, R);
        }
    }
}

bool Symmetry::novo(const dmat33& L,
                    const dmat33& R) const
{
    // check whether L and R are both identity matrix or not
    dmat33 I = dmat33::Identity();
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
        dmat33 L = _L[i] * _L[j];
        dmat33 R = _R[i] * _R[j];

        if (novo(L, R))
        {
            append(L, R);

            dvec4 quat;
            quaternion(quat, R);

            append(quat);

            umat tableNew = umat::Zero(table.rows() + 1,
                                       table.cols() + 1);

            tableNew.topLeftCorner(table.rows(), table.cols()) = table;

            table = tableNew;
        }
    }
}

void display(const Symmetry& sym)
{
    dmat33 L, R;

    for (int i = 0; i < sym.nSymmetryElement(); i++)
    {
        printf("%03d Symmetry Element:\n", i);

        sym.get(L, R, i);

        std::cout << "L matrix:\n" << L << std::endl << std::endl;
        std::cout << "R matrix:\n" << R << std::endl << std::endl;

        std::cout << "Quaternion:\n" << sym.quat(i) << std::endl << std::endl;

        rotate3D(R, sym.quat(i));
        std::cout << "R' matrix:\n" << R << std::endl << std::endl;
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

void symmetryCounterpart(dvec4& dst,
                         const Symmetry& sym)
{
    dvec4 q = dst;

    RFLOAT s = fabs(dst.dot(ANCHOR_POINT_2));

    dvec4 p;

    for (int i = 0; i < sym.nSymmetryElement(); i++)
    {
        quaternion_mul(p, quaternion_conj(sym.quat(i)), dst);

        RFLOAT t = fabs(p.dot(ANCHOR_POINT_2));

        if (t > s)
        {
            s = t;
            q = p;
        }
    }

    dst = q;
}

void symmetryRotation(vector<dmat33>& sr,
                      const dmat33 rot,
                      const Symmetry* sym)
{
    sr.clear();

    sr.push_back(rot);

    if (sym == NULL) return;

    if (asymmetry(*sym)) return;

    dmat33 L, R;

    for (int i = 0; i < sym->nSymmetryElement(); i++)
    {
        sym->get(L, R, i);
        
        sr.push_back(R * rot);
    }
}
