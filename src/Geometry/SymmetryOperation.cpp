/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#include "SymmetryOperation.h"

RotationSO::RotationSO(const int fold,
                       const double x,
                       const double y,
                       const double z)
{
    this->fold = fold;

    axis.resize(3);
    axis.set(x, 0);
    axis.set(y, 1);
    axis.set(z, 2);
}

ReflexionSO::ReflexionSO(const double x,
                         const double y,
                         const double z)
{
    plane.resize(3);
    plane.set(x, 0);
    plane.set(y, 1);
    plane.set(z, 2);
}

InversionSO::InversionSO()
{
}

SymmetryOperation::SymmetryOperation(const RotationSO rts)
{
    id = 0;

    fold = rts.fold;
    axisPlane = rts.axis;
}

SymmetryOperation::SymmetryOperation(const ReflexionSO rfs)
{
    id = 1;
    axisPlane = rfs.plane;
}

SymmetryOperation::SymmetryOperation(const InversionSO ivs)
{
    id = 2;

    axisPlane.resize(3);
}

void display(const SymmetryOperation so)
{
    switch (so.id)
    {
        case 0:
            printf("Rotation: fold = %2d, axis = %12.6f %12.6f %12.6f\n",
                   so.fold,
                   so.axisPlane.get(0),
                   so.axisPlane.get(1),
                   so.axisPlane.get(2));
            break;

        case 1:
            printf("Reflexion:   mirror plane = %12.6f %12.6f %12.6f\n",
                   so.axisPlane.get(0),
                   so.axisPlane.get(1),
                   so.axisPlane.get(2));
            break;

        case 2:
            printf("Inversion\n");
            break;
    }
}

void display(const vector<SymmetryOperation>& soList)
{
    for (size_t i = 0; i < soList.size(); i++)
        display(soList[i]);
}
