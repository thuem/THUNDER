#!/usr/bin/env python
#
# author Qiang Zhou
# author Zhao Wang
# author Shouqing Li
# author Mingxu Hu
# 
# version 1.4.11.081101
# copyright THUNDER Non-Commercial Software License Agreement
#
# ChangeLog
# AUTHOR      | TIME       | VERSION       | DESCRIPTION
# ------      | ----       | -------       | -----------
# Qiang  Zhou | 2015/03/23 | 0.0.1.050323  | new file
# Zhao   Wang | 2018/10/15 | 1.4.11.081015 | Euler transformation
# Shouqing Li | 2018/10/25 | 1.4.11.081025 | change method and option
# Mingxu   Hu | 2018/11/01 | 1.4.11.081101 | modify mistakes
#
# THU_2_STAR.py can translate a THU file form THUNDER into a STAR file used for RELION.

import os,sys
import random
import math
import numpy as np
from optparse import OptionParser

metalabels="_rlnVoltage \
_rlnDefocusU \
_rlnDefocusV \
_rlnDefocusAngle \
_rlnSphericalAberration \
_rlnAmplitudeContrast \
_rlnPhaseShift \
_rlnImageName \
_rlnMicrographName \
_rlnCoordinateX \
_rlnCoordinateY \
_rlnGroupNumber \
_rlnClassNumber \
_rlnAngleRot \
_rlnAngleTilt \
_rlnAnglePsi \
_rlnOriginX \
_rlnOriginY"

def SGN(x):
    y = 1 if x >= 0 else -1
    return y

def quaternion_to_euler(src):

    FLT_EPSILON = 1.19209*10**(-7)

    # mat : rotation matrix
    A = np.zeros((3,3))
    A[0,1] = -src[3]
    A[1,0] =  src[3]
    A[0,2] =  src[2]
    A[2,0] = -src[2]
    A[1,2] = -src[1]
    A[2,1] =  src[1]

    mat = np.eye(3) + 2 * src[0] * A + 2 * np.dot(A, A)
    mat = np.transpose(mat)

    if (abs(mat[1, 1]) > FLT_EPSILON):
        abs_sb = math.sqrt((-mat[2, 2] * mat[1, 2] * mat[2, 1] \
               - mat[0, 2] * mat[2, 0]) / mat[1, 1])
    elif (abs(mat[0, 1]) > FLT_EPSILON):
        abs_sb = math.sqrt((-mat[2, 1] * mat[2, 2] * mat[0, 2] + mat[2, 0] * mat[1, 2]) / mat[0, 1])
    elif (abs(mat[0, 0]) > FLT_EPSILON):
        abs_sb = math.sqrt((-mat[2, 0] * mat[2, 2] * mat[0, 2] - mat[2, 1] * mat[1, 2]) / mat[0, 0])
    else:
        print "Don't know how to extract angles."
        exit()

    if (abs_sb > FLT_EPSILON):
        beta  = math.atan2(abs_sb, mat[2, 2])
        alpha = math.atan2(mat[2, 1] / abs_sb, mat[2, 0] / abs_sb)
        gamma = math.atan2(mat[1, 2] / abs_sb, -mat[0, 2] / abs_sb)
    else:    
        alpha = 0
        beta  = 0
        gamma = math.atan2(mat[1, 0], mat[0, 0])

    gamma = math.degrees(gamma)
    beta  = math.degrees(beta)
    alpha = math.degrees(alpha)

    return (alpha, beta, gamma)

def main():

    prog_name = os.path.basename(sys.argv[0])
    usage = """
    Transform THU file to STAR file.
    {prog} < -i input_thu file> < -o output_star file > 
    """.format(prog = prog_name)

    optParser = OptionParser(usage)
    optParser.add_option("-i", \
                         "--input", \
                         action = "store", \
                         type = "string", \
                         dest = "input_thu", \
                         help = "Input THUNDER data.thu file.")
    optParser.add_option("-o", \
                         "--output", \
                         action = "store", \
                         type = "string", \
                         dest = "output_star", \
                         help = "Output RELION data.star file.")
    (options, args) = optParser.parse_args()

    if len(sys.argv) == 1:
        print usage
        print "    For more detail, see '-h' or '--help'."
    
    if options.output_star:
        fout = open(options.output_star, "w")
        fout.write("\ndata_\n\nloop_\n")
        n = 1
        for label in metalabels.split():
            fout.write("%s #%d\n"%(label, n))
            n += 1

        try:
            fin = open(options.input_thu, "r")
        except:
            print "Please input a proper thu file."
            exit()
        for line in fin:
            s = line.split()

            # voltage
            s[0] = str(float(s[0]) / 1000) # from V to kV

            # s[24] is Defocus value correction coefficient
            if len(s) > 20 and not s[24] == 0.0 :
                s[1] = s[1] # defocusU
                s[2] = s[2] # defocusV
                
            s[3] = str(float(s[3]) * 180. / math.pi ) # defocus angle, from rad to degree
            s[4] = str(float(s[4]) / 10000000. )      # Cs, from Angstrom to mm
            s[6] = str(float(s[6]) * 180. / math.pi ) # phase shift, from rad to degree

            for i in range(13,17):
                s[i] = float(s[i])
            quaternion = s[13:17] # quaternion for angles
            euler = quaternion_to_euler(quaternion)

            s[20] = str(float(s[20]) * -1.) # translation X
            s[21] = str(float(s[21]) * -1.) # translation Y

            sout = s[:12] + [str(int(s[12]) + 1), str(euler[0]), str(euler[1]), str(euler[2])] + s[20:22]

            fout.write(" ".join(sout) + "\n")

        fin.close()
        fout.close()

if __name__ == "__main__":
    main()
    
