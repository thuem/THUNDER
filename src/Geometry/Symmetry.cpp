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

Symmetry::Symmetry() {}

Symmetry::Symmetry(const char sym[])
{
    init(sym);
}

Symmetry::Symmetry(const int pgGroup,
                   const int pgOrder)
{
    init(pgGroup, pgOrder);
}

Symmetry::Symmetry(const vector<SymmetryOperation>& entry)
{
    init(entry);
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

    mat33 L, R;
    for (size_t i = 0; i < that.nSymmetryElement(); i++)
    {
        that.get(L, R, i);
        append(L, R);
    }

    return *this;
}

void Symmetry::get(mat33& L,
                   mat33& R,
                   const int i) const
{
    L = _L[i];
    R = _R[i];
}

void Symmetry::init(const char sym[])
{
    int pgGroup, pgOrder;
    symmetryGroup(pgGroup, pgOrder, sym);

    init(pgGroup, pgOrder);
}

void Symmetry::init(const int pgGroup,
                    const int pgOrder)
{
    vector<SymmetryOperation> entry;
    fillSymmetryEntry(entry, pgGroup, pgOrder);

    init(entry);
}

void Symmetry::init(const vector<SymmetryOperation>& entry)
{
    _L.clear();
    _R.clear();
    
    fillLR(entry);
    completePointGroup();
}

int Symmetry::nSymmetryElement() const
{
    return _L.size();
}

void Symmetry::clear()
{
    _L.clear();
    _R.clear();
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
        L.eye();

        if (entry[i].id == 0)
        {
            // printf("fold = %d\n", entry[i].fold);
            // rotation
            double angle = 2 * M_PI / entry[i].fold;
            // printf("angle = %f\n", angle);
            for (int j = 1; j < entry[i].fold; j++)
            {
                /***
                printf("angle * j = %f\n", angle * j);
                entry[i].axisPlane.print();
                ***/
                rotate3D(R, angle * j, entry[i].axisPlane);
                append(L, R);
            }
        }
        else if (entry[i].id == 1)
        {
            // reflexion
            reflect3D(R, entry[i].axisPlane);
            append(L, R);
        }
        else if (entry[i].id == 2)
        {
            /* inversion
             * L -> [ 1  0  0]
             *      [ 0  1  0]
             *      [ 0  0 -1]
             * R -> [-1  0  0]
             *      [ 0 -1  0]
             *      [ 0  0 -1] */
            L(2, 2) = -1;
            R.zeros();
            R.diag() = vec({-1, -1, -1});
            append(L, R);
        }
    }
}

bool Symmetry::novo(const mat33& L,
                    const mat33& R) const
{
    // check whether L and R are both identity matrix or not
    mat33 I(fill::eye);
    if (SAME_MATRIX(L, I) && SAME_MATRIX(R, I))
        return false;

    // check whether (L, R) exists in (_L, _R) or not
    for (int i = 0; i < _L.size(); i++)
        if (SAME_MATRIX(L, _L[i]) && SAME_MATRIX(R, _R[i]))
                return false;

    return true;
}

void Symmetry::completePointGroup()
{
    umat table(nSymmetryElement(),
               nSymmetryElement(),
               fill::zeros);

    int i, j;
    while ([&]
           {
                for (int row = 0; row < table.n_rows; row++)
                    for (int col = 0; col < table.n_cols; col++)
                        if (table(row, col) == 0)
                        {
                            i = row;
                            j = col;
                            table(row, col) = 1;
                    
                            return true;
                        }
                return false;
           }())

    {
        mat33 L = _L[i] * _L[j];
        mat33 R = _R[i] * _R[j];

        if (novo(L, R))
        {
            append(L, R);
            table.resize(table.n_rows + 1,
                         table.n_cols + 1);
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

        L.print("L matrix");
        R.print("R matrix");
    }
}
