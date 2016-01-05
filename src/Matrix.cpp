/*******************************************************************************
 * Author: Mingxu Hu, Gaochao Liu
 * Dependency:
 * Test:
 * Execution:
 * Description: a matrix class
 * ****************************************************************************/

#include "Matrix.h"

using namespace std;

template<class T>
Vector<T>::Vector() : _data(NULL)
{
}

template<class T>
Vector<T>::Vector(const T* data, const int length) : _data(NULL)
{
    init(length);

    memcpy(_data, data, length * sizeof(T));
}

template<class T>
Vector<T>::Vector(const int length) : _data(NULL)
{
    init(length);
}

template<class T>
Vector<T>::Vector(const Vector<T>& that)
{
    *this = that;
}

template<class T>
Vector<T>::~Vector()
{
    clear();
}

template<class T>
void Vector<T>::display() const
{
    cout << "It is a " << _length << " elements long vector." << endl;
    for (int i = 0; i < _length; i++)
    {
        cout.setf(ios::fixed);
        cout << setprecision(4) << _data[i] << " ";
    }
    cout << endl;
}

template<class T>
T Vector<T>::get(const int i) const
{
    return _data[i];
}

template<class T>
void Vector<T>::resize(const int length)
{
    clear();

    init(length);
}

template<class T>
void Vector<T>::zeros()
{
    for (int i = 0; i < length(); i++)
        _data[i] = T(0);
}

template<class T>
void Vector<T>::ones()
{
    for (int i = 0; i < length(); i++)
        _data[i] = T(1);
}

template<class T>
void Vector<T>::set(T value, int i)
{
    _data[i] = value;
}

template<class T>
int Vector<T>::length() const
{
    return _length;
}

template<class T>
T Vector<T>::dot(const Vector<T>& that) const
{
    if (this->length() != that.length())
        REPORT_ERROR("Can not dot two vectors not equally long.");
    
    T dotProduct = 0;
    for (int i = 0; i < _length; i++)
        dotProduct += _data[i] * that.get(i);

    return dotProduct;
}

template<class T>
T Vector<T>::modulusSquare() const
{
    T result = 0;

    for (int i = 0; i < _length; i++)
        result += _data[i] * _data[i];

    return result;
}

template<class T>
T Vector<T>::modulus() const
{
    return sqrt(modulusSquare());
}

template<class T>
void Vector<T>::clear()
{
    if (_data != NULL)
    {
        delete[] _data;
        _data = NULL;
    }
}

template<class T>
void Vector<T>::init(const int length)
{
    dataTypeCheck();

    if (length <= 0)
        REPORT_ERROR("Improper size for initializing a vector.");
         
    _data = new T[length];

    if (_data == NULL)
        REPORT_ERROR("Fail to allocate memory space for a vector.");

    _length = length;
}

template<class T>
void Vector<T>::dataTypeCheck() const
{
    if ((typeid(T) != typeid(int)) && 
        (typeid(T) != typeid(unsigned int)) &&
        (typeid(T) != typeid(long)) &&
        (typeid(T) != typeid(unsigned long)) &&
        (typeid(T) != typeid(float)) &&
        (typeid(T) != typeid(double)))
        REPORT_ERROR("Data type can not be accepted by a vector.");
}

template<class T>
Matrix<T>::Matrix() : _data(NULL)
{
}

template<class T>
Matrix<T>::Matrix(const T* data,
                  const int nRow,
                  const int nColumn) : _data(NULL)
{
    init(nRow, nColumn);

    memcpy(_data, data, nRow * nColumn * sizeof(T));
}

template<class T>
Matrix<T>::Matrix(const int nRow,
                  const int nColumn) : _data(NULL)
{
    init(nRow, nColumn);
}

template<class T>
Matrix<T>::Matrix(const Matrix<T>& that)
{ 
    *this = that;
}

template<class T>
Matrix<T>::~Matrix()
{
    clear();
}

template<class T>
void Matrix<T>::display() const
{
    cout << "It is a " << _nRow << " X " << _nColumn << " matrix." << endl;
    for (int i = 0; i < _nRow; i++)
    {
        int start = i * _nRow;
        for (int j = 0; j < _nColumn; j++)
        {
            cout.setf(ios::fixed);
            cout << setprecision(4) << _data[start++] << " ";
        }
        cout << endl;
    }
}

template<class T>
T Matrix<T>::get(const int i, const int j) const
{
    boundaryCheck(i, j); // check whether i and j fall in the matrix

    return _data[i * _nColumn + j];
}

template<class T>
T Matrix<T>::get(const int i) const
{
    if (i >= size())
        REPORT_ERROR("Out of the index boundary of the matrix.");

    return _data[i];
}

template<class T>
T* Matrix<T>::getData()
{
    return _data;
}

template<class T>
void Matrix<T>::resize(const int nRow, const int nColumn)
{
    clear();

    init(nRow, nColumn);
}

template<class T>
void Matrix<T>::zeros()
{
    for (int i = 0; i < size(); i++)
        _data[i] = T(0);
}

template<class T>
void Matrix<T>::ones()
{
    for (int i = 0; i < size(); i++)
        _data[i] = T(1); 
}

template<class T>
void Matrix<T>::identity()
{
    zeros();

    if (_nRow != _nColumn)
        REPORT_ERROR("Improper size for setting it an identity matrix.");

    for (int i = 0; i < _nRow; i++)
        set(1, i, i);
}

template<class T>
void Matrix<T>::set(T value)
{
    for (int i = 0; i < size(); i++)
        _data[i] = value;
}

template<class T>
void Matrix<T>::set(T value, const int i, const int j)
{
    boundaryCheck(i, j); // check whether i and j fall in the matrix

    _data[i * _nColumn + j] = value;
}

template<class T>
int Matrix<T>::nRow() const
{
    return _nRow;
}

template<class T>
int Matrix<T>::nColumn() const
{
    return _nColumn;
}

template<class T>
int Matrix<T>::size() const
{
    return _nRow * _nColumn;
}

template<class T>
Matrix<T> Matrix<T>::add(const Matrix<T>& that) const
{
    sizeCheck(that);
    
    Matrix<T> mat(this->nRow(), this->nColumn());

    for (int i = 0; i < this->nRow(); i++)
        for (int j = 0; j < this->nColumn(); j++)
            mat.set(this->get(i, j) + that.get(i, j), i, j);

    return mat;
}

template<class T>
Matrix<T> Matrix<T>::add(const T& value) const
{
    Matrix<T> mat(this->nRow(), this->nColumn());

    for (int i = 0; i < this->size(); i++)
        mat.getData()[i] = _data[i] + value;

    return mat;
}

template<class T>
Matrix<T> Matrix<T>::minus(const Matrix<T>& that) const
{
    sizeCheck(that);
    
    Matrix<T> mat(this->nRow(), this->nColumn());

    for (int i = 0; i < this->nRow(); i++)
        for (int j = 0; j < this->nColumn(); j++)
            mat.set(this->get(i, j) - that.get(i, j), i, j);

    return mat;
}

template<class T>
Matrix<T> Matrix<T>::minus(const T& value) const
{
    return add(-value);
}

template<class T>
Matrix<T> Matrix<T>::dot(const Matrix<T>& that) const
{
    sizeCheck(that);

    Matrix<T> mat(this->nRow(), this->nColumn());

    for (int i = 0; i < this->nRow(); i++)
        for (int j = 0; j < this->nColumn(); j++)
            mat.set(this->get(i, j) * that.get(i, j), i, j);

    return mat;
}

template<class T>
Matrix<T> Matrix<T>::dot(const T& that) const
{
    Matrix<T> mat(this->nRow(), this->nColumn());

    for (int i = 0; i < this->nRow(); i++)
        for (int j = 0; j < this->nColumn(); j++)
            mat.set(this->get(i, j) * that, i, j);
    
    return mat;
}

template<class T>
Matrix<T> Matrix<T>::multiply(const Matrix<T>& that) const
{
    sizeCheckMulti(that);

    Matrix<T> mat(this->nRow(), that.nColumn());

    for (int i = 0; i < this->nRow(); i++)
        for (int j = 0; j < that.nColumn(); j++)
        {
            T sum = 0;
            for (int k = 0; k < this->nColumn(); k++)
                sum += this->get(i, k) * that.get(k, j);
            mat.set(sum, i, j);
        }
        
    return mat;
}

template<class T>
Vector<T> Matrix<T>::multiply(const Vector<T>& that) const
{
    sizeCheckMulti(that);

    Vector<T> vec(this->nRow());

    for (int i = 0; i < this->nRow(); i++)
    {
        T sum = 0;
        for (int j = 0; j < that.length(); j++)
            sum += this->get(i, j) * that.get(j);
        vec.set(sum, i);
    }

    return vec;
}

template<class T>
Matrix<T> Matrix<T>::transpose() const
{
    Matrix<T> mat(this->nColumn(), this->nRow());
    for (int i = 0; i < this->nRow(); i++)
        for (int j = 0; j < this->nColumn(); j++)
            mat.set(this->get(i, j), j, i);
    return mat;
}

template<class T>
void Matrix<T>::replace(const Matrix<T>& mat,
                        const int i,
                        const int j)
{
    boundaryCheck(i, j);
    boundaryCheck(i + mat.nRow(), j + mat.nColumn());

    for (int rowIdx = 0; rowIdx < mat.nRow(); rowIdx++)
        for (int columnIdx = 0; columnIdx < mat.nColumn(); columnIdx++)
        {
            set(mat.get(rowIdx, columnIdx),
                i + rowIdx,
                j + columnIdx);
        }
}

template<class T>
void Matrix<T>::submat(Matrix<T>& mat,
                       const int i,
                       const int j) const
{
    boundaryCheck(i, j);
    boundaryCheck(i + mat.nRow(), j + mat.nColumn());
       
    for (int rowIdx = 0; rowIdx < mat.nRow(); rowIdx++)
        for (int columnIdx = 0; columnIdx < mat.nColumn(); columnIdx++)
        {
            mat.set(get(i + rowIdx, j + columnIdx),
                    rowIdx,
                    columnIdx);
        }
}

template<class T>
bool Matrix<T>::equal(const Matrix<T>& mat,
                      const float threshold) const
{
    for (int i = 0; i < _nRow; i++)
        for (int j = 0; j < _nColumn; j++)
            if (abs(get(i, j) - mat.get(i, j)) > threshold)
                return false;

    return true;
}

template<class T>
void Matrix<T>::clear()
{
    if (_data != NULL)
    {
        delete[] _data;
        _data = NULL;
    }
}

template<class T>
void Matrix<T>::init(const int nRow,
                     const int nColumn)
{
    dataTypeCheck();

    if (nRow <= 0 || nColumn <= 0)
        REPORT_ERROR("Improper size for initializing a matrix.");
     
    _data = new T[nRow * nColumn];

    if (_data == NULL)
        REPORT_ERROR("Fail to allocate memory space for a matrix.");

    _nRow = nRow;
    _nColumn = nColumn;
}

template<class T>
void Matrix<T>::dataTypeCheck() const
{
    if ((typeid(T) != typeid(int)) && 
        (typeid(T) != typeid(unsigned int)) &&
        (typeid(T) != typeid(long)) &&
        (typeid(T) != typeid(unsigned long)) &&
        (typeid(T) != typeid(float)) &&
        (typeid(T) != typeid(double)))
        REPORT_ERROR("Data type can not be accepted by a matrix.");
}

template<class T>
void Matrix<T>::boundaryCheck(int i, int j) const
{
    if ((i < 0) || (i >= _nRow) || (j < 0) || (j >= _nColumn))
        REPORT_ERROR("Out of the boundary of the matrix.");
}

template<class T>
void Matrix<T>::sizeCheck(const Matrix<T>& that) const
{
    if (this->nRow() != that.nRow() || this->nColumn() != that.nColumn())
        REPORT_ERROR("The operation needs the two matrix have same sizes.");
}

template<class T>
void Matrix<T>::sizeCheckMulti(const Matrix<T>& that) const
{
    if (this->nColumn() != that.nRow())
    {
        printf("nColumn = %d\n", this->nColumn());
        printf("nRow = %d\n", that.nRow());
        REPORT_ERROR("The size demand of mulitpy two matrixes is not \
                      fullfilled.");
    }
}

template<class T>
void Matrix<T>::sizeCheckMulti(const Vector<T>& that) const
{
    if (this->nColumn() != that.length())
        REPORT_ERROR("The size demand of mulitpy a matrix and a vector is not \
                      fullfilled.");
}
