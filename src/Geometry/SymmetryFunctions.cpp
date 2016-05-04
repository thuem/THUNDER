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
#include "Utils.h"

void symmetryGroup(int& pgGroup,
                   int& pgOrder,
                   const char sym[])
{
    if (regexMatches(sym, "^C[[:digit:]]+$"))
    {
        pgGroup = PG_CN;
        pgOrder = atoi(&sym[1]);
    }
    else if (strcmp(sym, "CI") == 0)
    {
        pgGroup = PG_CI;
        pgOrder = -1;
    }
    else if (strcmp(sym, "CS") == 0)
    {
        pgGroup = PG_CS;
        pgOrder = -1;
    }
    else if (regexMatches(sym, "^C[[:digit:]]+H$"))
    {
        pgGroup = PG_CNH;
        pgOrder = atoi(&sym[1]);
    }
    else if (regexMatches(sym, "^C[[:digit:]]+V$"))
    {
        pgGroup = PG_CNV;
        pgOrder = atoi(&sym[1]);
    }
    else if (regexMatches(sym, "^S[[:digit:]]+$"))
    {
        pgGroup = PG_SN;
        pgOrder = atoi(&sym[1]);
    }
    else if (regexMatches(sym, "^D[[:digit:]]+$"))
    {
        pgGroup = PG_DN;
        pgOrder = atoi(&sym[1]);
    }
    else if (regexMatches(sym, "^D[[:digit:]]+H$"))
    {
        pgGroup = PG_DNH;
        pgOrder = atoi(&sym[1]);
    }
    else if (regexMatches(sym, "^D[[:digit:]]+V$"))
    {
        pgGroup = PG_DNV;
        pgOrder = atoi(&sym[1]);
    }
    else if (strcmp(sym, "T") == 0)
    {
        pgGroup = PG_T;
        pgOrder = -1;
    }
    else if (strcmp(sym, "TD") == 0)
    {
        pgGroup = PG_TD;
        pgOrder = -1;
    }
    else if (strcmp(sym, "TH") == 0)
    {
        pgGroup = PG_TH;
        pgOrder = -1;
    }
    else if (strcmp(sym, "O") == 0)
    {
        pgGroup = PG_O;
        pgOrder = -1;
    }
    else if (strcmp(sym, "OH") == 0)
    {
        pgGroup = PG_OH;
        pgOrder = -1;
    }
    else if (strcmp(sym, "I") == 0)
    {
        pgGroup = PG_I;
        pgOrder = -1;
    }
    else if (regexMatches(sym, "^I[12345]$"))
    {
        switch (sym[1])
        {
            case '1':
                pgGroup = PG_I1; break;
            case '2':
                pgGroup = PG_I2; break;
            case '3':
                pgGroup = PG_I3; break;
            case '4':
                pgGroup = PG_I4; break;
            case '5':
                pgGroup = PG_I5; break;
        }
        pgOrder = -1;
    }
    else if (strcmp(sym, "IH") == 0)
    {
        pgGroup = PG_IH;
        pgOrder = -1;
    }
    else if (regexMatches(sym, "^I[12345]H$"))
    {
        switch (sym[1])
        {
            case '1':
                pgGroup = PG_I1H; break;
            case '2':
                pgGroup = PG_I2H; break;
            case '3':
                pgGroup = PG_I3H; break;
            case '4':
                pgGroup = PG_I4H; break;
            case '5':
                pgGroup = PG_I5H; break;
        }
        pgOrder = -1;
    }
    else
        LOG(FATAL) << "Invalid Symmetry Index";
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

        case PG_CI:
            entry.push_back(SymmetryOperation(InversionSO()));
            break;

        case PG_CS:
            entry.push_back(SymmetryOperation(ReflexionSO(0, 0, 1)));
            break;

        case PG_CNV:
            fillSymmetryEntry(entry, PG_CN, pgOrder);
            entry.push_back(SymmetryOperation(ReflexionSO(0, 1, 0)));
            break;

        case PG_CNH:
            fillSymmetryEntry(entry, PG_CN, pgOrder);
            entry.push_back(SymmetryOperation(ReflexionSO(0, 0, 1)));
            break;

        case PG_SN:
            if (pgOrder % 2 == 1)
                REPORT_ERROR("order for SN group must be even");
            fillSymmetryEntry(entry, PG_CN, pgOrder / 2);
            entry.push_back(SymmetryOperation(InversionSO()));
            break;

        case PG_DN:
            fillSymmetryEntry(entry, PG_CN, pgOrder);
            entry.push_back(SymmetryOperation(RotationSO(2, 1, 0, 0)));
            break;

        case PG_DNV:
            fillSymmetryEntry(entry, PG_DN, pgOrder);
            entry.push_back(SymmetryOperation(ReflexionSO(1, 0, 0)));
            break;

        case PG_DNH:
            fillSymmetryEntry(entry, PG_DN, pgOrder);
            entry.push_back(SymmetryOperation(ReflexionSO(0, 0, 1)));
            break;

        case PG_T:
            entry.push_back(SymmetryOperation(RotationSO(3, 0, 0, 1)));
            entry.push_back(SymmetryOperation(RotationSO(2,
                                                         0,
                                                         0.816496,
                                                         0.577350)));
            break;

        case PG_TD:
            fillSymmetryEntry(entry, PG_T, pgOrder);
            entry.push_back(SymmetryOperation(ReflexionSO(1.4142136,
                                                          2.4494897,
                                                          0)));
            break;

        case PG_TH:
            entry.push_back(SymmetryOperation(RotationSO(3, 0, 0, 1)));
            entry.push_back(SymmetryOperation(RotationSO(2,
                                                         0,
                                                         -0.816496,
                                                         -0.577350)));
            entry.push_back(SymmetryOperation(InversionSO()));
            break;

        case PG_O:
            entry.push_back(SymmetryOperation(RotationSO(3,
                                                         0.5773502,
                                                         0.5773502,
                                                         0.5773502)));
            entry.push_back(SymmetryOperation(RotationSO(4, 0, 0, 1)));
            break;

        case PG_OH:
            fillSymmetryEntry(entry, PG_O, pgOrder);
            entry.push_back(SymmetryOperation(ReflexionSO(0, 1, 1)));
            break;

        case PG_I:
        case PG_I2:
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

        case PG_I1:
            fillSymmetryEntry(entry, PG_CN, 2);
            entry.push_back(SymmetryOperation(RotationSO(5,
                                                         0.8506508,
                                                         0,
                                                         -0.5257311)));
            entry.push_back(SymmetryOperation(RotationSO(3,
                                                         0.9341724,
                                                         0.3568221,
                                                         0)));
            break;

        case PG_I3:
            entry.push_back(SymmetryOperation(RotationSO(2,
                                                         -0.5257311,
                                                         0,
                                                         0.8506508)));
            fillSymmetryEntry(entry, PG_CN, 5);
            entry.push_back(SymmetryOperation(RotationSO(3,
                                                         -0.4911235,
                                                         0.3568221,
                                                         0.7946545)));
            break;

        case PG_I4:
            entry.push_back(SymmetryOperation(RotationSO(2,
                                                         0.5257311,
                                                         0,
                                                         0.8506508)));
            entry.push_back(SymmetryOperation(RotationSO(5,
                                                         0.8944272,
                                                         0,
                                                         0.4472136)));
            entry.push_back(SymmetryOperation(RotationSO(3,
                                                         0.4911235,
                                                         0.3568221,
                                                         0.7946545)));
            break;

        case PG_I5:
            LOG(FATAL) << "PG_I5 Not Implemented";
            break;

        case PG_IH:
        case PG_I2H:
            fillSymmetryEntry(entry, PG_I, pgOrder);
            entry.push_back(SymmetryOperation(ReflexionSO(1, 0, 0)));
            break;

        case PG_I1H:
            fillSymmetryEntry(entry, PG_I1, pgOrder);
            entry.push_back(SymmetryOperation(ReflexionSO(0, 0, -1)));
            break;

        case PG_I3H:
            fillSymmetryEntry(entry, PG_I3, pgOrder);
            entry.push_back(SymmetryOperation(ReflexionSO(0.8506508,
                                                          0,
                                                          0.5257311)));
            break;
        
        case PG_I4H:
            fillSymmetryEntry(entry, PG_I4, pgOrder);
            entry.push_back(SymmetryOperation(ReflexionSO(0.8506508,
                                                          0,
                                                          -0.5257311)));
            break;

        case PG_I5H:
            LOG(FATAL) << "PG_I5H Not Implemented";
            break;

        default:
            LOG(FATAL) << "Symmetry Point Group is Not Known";
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
