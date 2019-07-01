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

#include <LinearAlgebra.h>

// INITIALIZE_EASYLOGGINGPP

class LinearAlegbraTest : public :: testing:: Test
{
    protected:

        void setUp()
        {
            _v1 = dvec4(1, 0, 0, 0);
            _v2 = dvec4(1, 0, 0, 0);
            _v3 = dvec4(0, 0, 0, 1);
        }

        dvec4 _v1;
        dvec4 _v2;
        dvec4 _v3;
};

TEST_F(LinearAlegbraTest, Equivalence_1)
{
    EXPECT_TRUE((_v1 == _v1));
    EXPECT_TRUE((_v1 == _v2));
    EXPECT_FALSE(_v1 == _v3);
    EXPECT_FALSE(_v2 == _v3);
}

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);

    // loggerInit(argc, argv);

    return RUN_ALL_TESTS();
}
