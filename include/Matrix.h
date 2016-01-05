/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#include <typeinfo>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <cmath>

#include "Typedef.h"
#include "Error.h"
#include "Macro.h"

#ifndef MATRIX_H
#define MATRIX_H

template<class T>
class Vector
{
    private:

        T* _data = NULL;

        int _length;

    public:

        Vector();

        Vector(const T* data, const int length);

        Vector(const int length);

        Vector(const Vector<T>& that);

        ~Vector(); 

        Vector<T>& operator=(const Vector<T>& that)
        {
            clear();

            init(that.length());

            for (int i = 0; i < _length; i++)
                _data[i] = that.get(i);

            return *this;
        }

        void display() const;

        T get(const int i) const;

        T& operator()(const int i)
        {
            return _data[i];
        }

        void resize(const int length);

        void zeros();

        void ones();
        
        void set(T value, int i);

        int length() const;

        T dot(const Vector<T>& that) const;
        friend T operator*(Vector<T>& left,
                           Vector<T>& right)
        {
            return left.dot(right);
        }

        friend void operator*=(Vector<T>& left, const T right)
        {
            for (int i = 0; i < left.length(); i++)
                left[i] *= right;
        }

        T modulusSquare() const;

        T modulus() const;

        void clear();

    private:

        void init(const int length);

        void dataTypeCheck() const;
};

template<class T>
class Matrix
{
    private:

        T* _data;
        // a pointer points to the data block
        // use row-major standard for storing matrix

        int _nRow; // the number of rows in this matrix
        int _nColumn; // the number of columns in this matrix

    public:

        Matrix();

        Matrix(const T* data,
               const int nRow,
               const int nColumn);

        Matrix(const int nRow,
               const int nColumn);

        Matrix(const Matrix<T>& that);

        ~Matrix();

        Matrix<T>& operator=(const Matrix<T>& that)
        {
            clear();

            init(that.nRow(), that.nColumn());

            for (int i = 0; i < _nRow * _nColumn; i++)
                _data[i] = that.get(i);

            return *this;
        }

        void display() const;

        T get(const int i, const int j) const;
        T get(const int i) const;

        T& operator[](const int i)
        {
            return _data[i];
        }

        T& operator()(const int i)
        {
            return _data[i];
        }

        T& operator()(const int i,
                      const int j)
        {
            return _data[i * _nColumn + j];
        }

        T* getData();
        // get a pointer which points to the memory space storing the matrix

        void resize(const int nRow, const int Column);

        void zeros();
        // make every element in this matrix be zero
        void ones();
        // make every element in this matrix be one
        void identity();
        // If (_nRow == _nColumn), make this matrix an identity matrix,
        // else throw out an error. 

        void set(T value);
        // make every element in this matrix be value
        void set(T value, const int i, const int j);

        int nRow() const;
        int nColumn() const;
        int size() const;

        Matrix<T> add(const Matrix<T>& that) const;
        friend Matrix<T> operator+(const Matrix<T>& left,
                                   const Matrix<T>& right)
        {
            return left.add(right);
        }

        Matrix<T> add(const T& value) const;
        friend Matrix<T> operator+(const Matrix<T>& left, const T& value)
        {
            return left.add(value);
        }
        friend Matrix<T> operator+(const T& value, const Matrix<T>& right)
        {
            return right.add(value);
        }

        Matrix<T> minus(const Matrix<T>& that) const;
        friend Matrix<T> operator-(const Matrix<T>& left,
                                   const Matrix<T>& right)
        {
            return left.minus(right);
        }

        Matrix<T> minus(const T& value) const;
        friend Matrix<T> operator-(const Matrix<T>& left, const T& value)
        {
            return left.minus(value);
        }

        Matrix<T> dot(const Matrix<T>& that) const; 

        Matrix<T> dot(const T& that) const;
        friend Matrix<T> operator*(const Matrix<T>& left,
                                   const T& right)
        {
            return left.dot(right);
        }

        Matrix<T> multiply(const Matrix<T>& that) const;
        friend Matrix<T> operator*(const Matrix<T>& left,
                                   const Matrix<T>& right)
        {
            return left.multiply(right);
        }

        Vector<T> multiply(const Vector<T>& that) const;
        friend Vector<T> operator*(const Matrix<T>& left,
                                   const Vector<T>& right)
        {
            return left.multiply(right);
        }

        Matrix<T> transpose() const;

        void replace(const Matrix<T>& mat,
                     const int i,
                     const int j);
        // replace the submatrix (i -> i + mat.nRow()
        //                        j -> j + mat.nColumn())
        // with mat
        
        void submat(Matrix<T>& mat,
                    const int i,
                    const int j) const;
        // fill mat with submatrix (i -> i + mat.nRow()
        //                          j -> j + mat.nColumn()
        // with mat
        
        bool equal(const Matrix<T>& mat,
                   const float threshold = EQUAL_ACCURACY) const;
        friend bool operator==(const Matrix<T>& left,
                               const Matrix<T>& right)
        {
            return left.equal(right);
        }

        void clear();

    private:

        void init(const int nRow,
                  const int nColumn);

        void dataTypeCheck() const;

        void boundaryCheck(int i, int j) const;

        void sizeCheck(const Matrix<T>& that) const;
        void sizeCheckMulti(const Matrix<T>& that) const;
        void sizeCheckMulti(const Vector<T>& that) const;
};

#include "../src/Matrix.cpp"

#endif // MATRIX_H
