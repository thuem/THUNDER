set(CPPUNIT_PATH "${PROJECT_BINARY_DIR}/external/cppunit")

set(ext_conf_flags_cppunit --prefix=${CPPUNIT_PATH})

set(CPPUNIT_LIBRARIES ${CPPUNIT_PATH}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}cppunit${CMAKE_STATIC_LIBRARY_SUFFIX})

message(STATUS "CPPUNIT_LIBRARIES : ${CPPUNIT_LIBRARIES}")

include(ExternalProject)

externalproject_add(CPPUNIT
                    SOURCE_DIR ${PROJECT_SOURCE_DIR}/external/packages/cppunit
                    CONFIGURE_COMMAND <SOURCE_DIR>/configure ${ext_conf_flags_cppunit}
                    INSTALL_DIR ${CPPUNIT_PATH}
                    BINARY_DIR ${CPPUNIT_PATH}/build
                    LOG_INSTALL)
