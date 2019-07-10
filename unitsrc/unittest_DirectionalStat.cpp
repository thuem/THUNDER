/** @file
 *  @author Mingxu Hu 
 *  @version 1.4.11.081102
 *  @copyright GPLv2
 *
 *  ChangeLog
 *  AUTHOR      | TIME       | VERSION       | DESCRIPTION
 *  ------      | ----       | -------       | -----------
 *  Mingxu Hu   | 2019/06/29 | 1.4.14.090629 | new file
 */

#include <gtest/gtest.h>

#include <DirectionalStat.h>

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

INITIALIZE_EASYLOGGINGPP

class DirectionalStatTest : public :: testing:: Test
{
    protected:

        void setUp()
        {
            _set0 = dmat4::Zero(100, 4);
            _set0.col(0) = dvec::Ones(100);
        }

        dmat4 _set0;
        dmat4 _set1;
        dmat4 _set2;
};

TEST_F(DirectionalStatTest, MeanOfStillRotations)
{
    dvec4 m;
    mean(m, _set0);

    // EXPECT_EQ(mean, dvec4(1, 0, 0, 0));
}

class DirectionalStatTestP : public :: testing:: TestWithParam<int>
{
};

TEST_P(DirectionalStatTestP, TEST)
{
}

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);

    loggerInit(argc, argv);

    return RUN_ALL_TESTS();
}
