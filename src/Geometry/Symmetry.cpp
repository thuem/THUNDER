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

Symmetry::Symmetry()
{
}

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

void Symmetry::operator=(const Symmetry& that)
{
    clear();

    Matrix<double> L(4, 4);
    Matrix<double> R(4, 4);
    for (size_t i = 0; i < that.nSymmetryElement(); i++)
    {
        that.get(L, R, i);
        append(L, R);
    }
}

void Symmetry::get(Matrix<double>& L,
                   Matrix<double>& R,
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

void Symmetry::append(const Matrix<double>& L,
                      const Matrix<double>& R)
{
    _L.push_back(L);
    _R.push_back(R);
}

void Symmetry::set(const Matrix<double>& L,
                   const Matrix<double>& R,
                   const int i)
{
    _L[i] = L;
    _R[i] = R;
}

void Symmetry::fillLR(const vector<SymmetryOperation>& entry)
{
    Matrix<double> L(4, 4);
    Matrix<double> R(4, 4);

    for (size_t i = 0; i < entry.size(); i++)
    {
        L.identity();

        if (entry[i].id == 0)
        {
            // rotation
            double angle = 2 * PI / entry[i].fold;
            for (int j = 1; j < entry[i].fold; j++)
            {
                rotate3D(R, angle * j, entry[i].axisPlane);
                homogenize(R);
                append(L, R);
            }
        }
        else if (entry[i].id == 1)
        {
            // reflexion
            reflect3D(R, entry[i].axisPlane);
            homogenize(R);
            append(L, R);
        }
        else if (entry[i].id == 2)
        {
            /* inversion
             * L -> [ 1  0  0  0]
             *      [ 0  1  0  0]
             *      [ 0  0 -1  0]
             *      [ 0  0  0  1]
             * R -> [-1  0  0  0]
             *      [ 0 -1  0  0]
             *      [ 0  0 -1  0]
             *      [ 0  0  0  1] */
            L.set(-1, 2, 2);
            R.identity();
            R.set(-1, 0, 0);
            R.set(-1, 1, 1);
            R.set(-1, 2, 2);
            append(L, R);
        }
    }
}

bool Symmetry::novo(const Matrix<double>& L,
                    const Matrix<double>& R) const
{
    // check whether L and R are both identity matrix or not
    Matrix<double> I(4, 4);
    I.identity();
    if ((L == I) && (R == I))
        return false;

    // check whether (L, R) exists in (_L, _R) or not
    for (int i = 0; i < _L.size(); i++)
        if ((L == _L[i]) && (R == _R[i]))
            return false;

    return true;
}

void Symmetry::completePointGroup()
{
    Matrix<int> table(nSymmetryElement(),
                      nSymmetryElement());
    table.zeros();

    int i, j;
    while ([&]
           {
                for (int row = 0; row < table.nRow(); row++)
                    for (int col = 0; col < table.nColumn(); col++)
                        if (table.get(row, col) == 0)
                        {
                            i = row;
                            j = col;
                            table.set(1, row, col);
                    
                            return true;
                        }
                return false;
           }())

    {
        Matrix<double> L = _L[i] * _L[j];
        Matrix<double> R = _R[i] * _R[j];

        if (novo(L, R))
        {
            append(L, R);
            [&]
            {
                Matrix<int> tmp = table;
                table.resize(table.nRow() + 1,
                             table.nColumn() + 1);
                table.zeros();
                table.replace(tmp, 0, 0);
            }();
        }
    }
}

void display(const Symmetry& sym)
{
    Matrix<double> L(4, 4);
    Matrix<double> R(4, 4);

    for (int i = 0; i < sym.nSymmetryElement(); i++)
    {
        printf("%03d Symmetry Element:\n", i);

        sym.get(L, R, i);

        printf("L matrix:\n");
        L.display();

        printf("R matrix:\n");
        R.display();

        printf("\n");
    }
}
