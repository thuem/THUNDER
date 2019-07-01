/** @file
 *  @author Mingxu Hu 
 *  @version 1.4.11.081102
 *  @copyright THUNDER Non-Commercial Software License Agreement
 *
 *  ChangeLog
 *  AUTHOR      | TIME       | VERSION       | DESCRIPTION
 *  ------      | ----       | -------       | -----------
 *  Mingxu Hu   | 2018/11/02 | 1.4.11.081102 | new file
 *
 *
 */

#include <gtest/gtest.h>

#include <Complex.h>

/***
TEST(TestComplexAdd, Test01)
{
    EXPECT_EQ(COMPLEX(0, 0) + COMPLEX(0, 0), COMPLEX(0, 0));
    EXPECT_EQ(COMPLEX(1, 0) + COMPLEX(0, 0), COMPLEX(1, 0));
    EXPECT_EQ(COMPLEX(0, 0) + COMPLEX(0, 1), COMPLEX(0, 1));
    EXPECT_EQ(COMPLEX(1, 0) + COMPLEX(0, 1), COMPLEX(1, 1));
}

TEST(TestComplexMinus, Test01)
{
    EXPECT_EQ(COMPLEX(0, 0) - COMPLEX(0, 0), COMPLEX(0, 0));
    EXPECT_EQ(COMPLEX(1, 0) - COMPLEX(0, 0), COMPLEX(1, 0));
    EXPECT_EQ(COMPLEX(0, 0) - COMPLEX(0, 1), COMPLEX(0, -1));
    EXPECT_EQ(COMPLEX(1, 0) - COMPLEX(0, 1), COMPLEX(1, -1));
}
***/

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
