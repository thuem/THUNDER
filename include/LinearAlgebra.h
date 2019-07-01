/** @file
 *  @author Mingxu Hu
 *  @version 1.4.14.190629
 *  @copyright GPLv2
 *
 *  ChangeLog
 *  AUTHOR     | TIME       | VERSION       | DESCRIPTION
 *  ------     | ----       | -------       | -----------
 *  Mingxu Hu  | 2019/06/29 | 1.4.14.190629 | new file
 *
 *  @brief 
 *
 */

#ifndef LINEAR_ALGEBRA_H
#define LINEAR_ALGEBRA_H

#include "THUNDERConfig.h"
#include "Macro.h"
#include "Precision.h"
#include "Typedef.h"

inline bool operator==(const dvec4& v1,
                       const dvec4& v2)
{
    for (int i = 0; i < 4; i++)
    {
        if (v1[i] != v2[i])
            return false;
    }

    return true;
}

#endif // LINEAR_ALGEBRA_H
