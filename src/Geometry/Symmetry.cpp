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

void Symmetry::get(mat33& L,
                   mat33& R,
                   const int i) const
{
    L = _L[i];
    R = _R[i];
}

vec4 Symmetry::quat(const int i) const
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

void Symmetry::append(const mat33& L,
                      const mat33& R)
{
    _L.push_back(L);
    _R.push_back(R);
}

void Symmetry::append(const vec4& quat)
{
    _quat.push_back(quat);
}

void Symmetry::set(const mat33& L,
                   const mat33& R,
                   const int i)
{
    _L[i] = L;
    _R[i] = R;
}

void Symmetry::set(const vec4& quat,
                   const int i)
{
    _quat[i] = quat;
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

                if (novo(L, R))
                {
                    append(L, R);

                    vec4 quat;
                    quaternion(quat, R);
                    append(quat);
                }

                /***
                if (novo(L, R.transpose()))
                {
                    append(L, R.transpose());

                    vec4 quat;
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

            vec4 quat;
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
    mat33 L, R;

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

void symmetryCounterpart(vec4& dst,
                         const Symmetry& sym)
{
    /***
    vector<SymmetryOperation> entry;

    fillSymmetryEntry(entry, sym.pgGroup(), sym.pgOrder());

    for (size_t i = 0; i < entry.size(); i++)
    {
        if (entry[i].id == 0)
        {
            double phi, theta, psi;

            angle(phi, theta, psi, dst);

            cout << "phi_1 = " << phi << endl;

            double d = 2 * M_PI / entry[i].fold;

            double t = vec3(dst(1), dst(2), dst(3)).dot(entry[i].axisPlane);

            double p = 2 * acos(dst(0) / sqrt(gsl_pow_2(dst(0)) + gsl_pow_2(t)));

            cout << "phi_2 = " << p << endl;

            // cout << "phi = " << phi * 180 / M_PI << endl;

            int n = periodic(p, d);

            // cout << "n = " << n << endl;

            if (n < 0) REPORT_ERROR("N WRONG!");

            for (int j = 0; j < n; j++)
                quaternion_mul(dst,
                               quaternion_conj(vec4(cos(d / 2),
                                                    entry[i].axisPlane(0) * sin(d / 2),
                                                    entry[i].axisPlane(1) * sin(d / 2),
                                                    entry[i].axisPlane(2) * sin(d / 2))),
                               dst);

            angle(phi, theta, psi, dst);

            cout << "phi_3 = " << phi << endl;
        }
        else
        {
            REPORT_ERROR("WRONG");
            abort();
        }
    }
    ***/

    /***
    double s = fabs(dst(0));

    int j = 0;

    for (int i = 0; i < sym.nSymmetryElement(); i++)
    {
        double t = fabs(dst.dot(sym.quat(i)));

        if (t > s)
        {
            s = t;

            j = i + 1;
        }
    }

    if (j != 0)
        quaternion_mul(dst, quaternion_conj(sym.quat(j - 1)), dst);
    ***/

    vec4 q = dst;

    double s = fabs(dst.dot(ANCHOR_POINT_2));

    vec4 p;

    for (int i = 0; i < sym.nSymmetryElement(); i++)
    {
        quaternion_mul(p, quaternion_conj(sym.quat(i)), dst);
        //quaternion_mul(p, sym.quat(i), dst);
        //quaternion_mul(p, dst, sym.quat(i));

        double t = fabs(p.dot(ANCHOR_POINT_2));

        if (t > s)
        {
            s = t;
            q = p;
        }
    }

    dst = q;

    /***
    mat4 anchors(sym.nSymmetryElement(), 4);

    vec4 anchor;
    for (int i = 0; i < sym.nSymmetryElement(); i++)
    {
        quaternion_mul(anchor, sym.quat(i), ANCHOR_POINT_1);
        quaternion_mul(anchor, anchor, quaternion_conj(sym.quat(i)));

        anchors.row(i) = anchor.transpose();
    }

    vec4 r;
    quaternion_mul(r, dst, ANCHOR_POINT_0);
    quaternion_mul(r, r, quaternion_conj(dst));

    int j = 0; // index of the nearest anchor
    double s = r.dot(ANCHOR_POINT_1);

    for (int i = 0; i < sym.nSymmetryElement(); i++)
    {
        double t = r.dot(anchors.row(i).transpose());

        if (t > s)
        {
            s = t;
            j = i + 1;
        }
    }

    if (j != 0)
    {
        dst(0) = 1;
        dst(1) = 0;
        dst(2) = 0;
        dst(3) = 0;

        //quaternion_mul(dst, quaternion_conj(sym.quat(j - 1)), dst);
    }
    ***/

    /***
    vec4 q = dst;

    vec4 r;
    quaternion_mul(r, q, ANCHOR_POINT_1);
    quaternion_mul(r, r, quaternion_conj(q));

    double s = r.dot(ANCHOR_POINT_1);

    vec4 p;

    for (int i = 0; i < sym.nSymmetryElement(); i++)
    {
        quaternion_mul(p, sym.quat(i), dst);

        quaternion_mul(r, p, ANCHOR_POINT_1);
        quaternion_mul(r, r, quaternion_conj(p));

        double t = r.dot(ANCHOR_POINT_1);

        if (t > s)
        {
            s = t;

            q = p;
        }
    }

    dst = q;
    ***/
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
