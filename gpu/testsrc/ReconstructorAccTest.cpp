#include "cuthunder.h"

#include <iostream>

#include "easylogging++.h"

_INITIALIZE_EASYLOGGINGPP

void test_class()
{
    LOG(INFO) << "add begin..";

    cuthunder::addTest();

    LOG(INFO) << "add done!";
}

int main(int argc, char *argv[])
{
    test_class();

    return 0;
}
