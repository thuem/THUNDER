/*******************************************************************************
 * Author: Kunpeng Wang, Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#include "SymmetryFunctions.h"

void symmetryGroup(int& pgGroup,
                   int& pgOrder,
                   const char sym[])
{
    if (regexMatches(sym, "^C[[:digit:]]+$"))
    {
        pgGroup = PG_CN;
        pgOrder = atoi(&sym[1]);
    }
    else if (regexMatches(sym, "^D[[:digit:]]+$"))
    {
        pgGroup = PG_DN;
        pgOrder = atoi(&sym[1]);
    }
    else if (strcmp(sym, "T") == 0)
    {
        pgGroup = PG_T;
        pgOrder = -1;
    }
    else if (strcmp(sym, "O") == 0)
    {
        pgGroup = PG_O;
        pgOrder = -1;
    }
    else if (strcmp(sym, "I") == 0)
    {
        pgGroup = PG_I;
        pgOrder = -1;
    }
    else
        REPORT_ERROR("INVALID SYMMTRY INDEX");
}

void fillSymmetryEntry(vector<SymmetryOperation>& entry,
                       const int pgGroup,
                       const int pgOrder)
{
    switch (pgGroup)
    {
        case PG_CN:
            entry.push_back(SymmetryOperation(RotationSO(pgOrder, 0, 0, 1)));
            break;

        case PG_DN:
            fillSymmetryEntry(entry, PG_CN, pgOrder);
            entry.push_back(SymmetryOperation(RotationSO(2, 1, 0, 0)));
            break;

        case PG_T:
            entry.push_back(SymmetryOperation(RotationSO(3, 0, 0, 1)));
            entry.push_back(SymmetryOperation(RotationSO(2,
                                                         0,
                                                         0.816496,
                                                         0.577350)));
            break;

        case PG_O:
            entry.push_back(SymmetryOperation(RotationSO(3,
                                                         0.5773502,
                                                         0.5773502,
                                                         0.5773502)));
            entry.push_back(SymmetryOperation(RotationSO(4, 0, 0, 1)));
            break;

        case PG_I:
            fillSymmetryEntry(entry, PG_CN, 2);
            entry.push_back(SymmetryOperation(RotationSO(5,
                                                         0.5257311,
                                                         0,
                                                         0.8506508)));
            entry.push_back(SymmetryOperation(RotationSO(3,
                                                         0,
                                                         0.3568221,
                                                         0.9341723)));
            break;

        default:
            REPORT_ERROR("UNKNOWN SYMMETRY POINT GROUP KNOWN");
            abort();
            break;
    }
}

void countSymmetryElement(int& nSymmetryElement,
                          int& nRotation,
                          int& nReflexion,
                          int& nInversion,
                          const vector<SymmetryOperation>& entry)
{
    nSymmetryElement = 0;
    nRotation = 0;
    nReflexion = 0;
    nInversion = 0;

    for (size_t i = 0; i < entry.size(); i++)
    {
        switch (entry[i].id)
        {
            case 0:
                nRotation += 1;
                nSymmetryElement += entry[i].fold - 1;
                break;

            case 1:
                nReflexion += 1;
                nSymmetryElement += 1;
                break;

            case 2:
                nInversion += 1;
                nSymmetryElement += 1;
                break;
        }
    }
}
