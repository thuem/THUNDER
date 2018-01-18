#!/usr/bin/env python

import math
import sys
import re

def main():

    fSTAR = sys.argv[1]

    header_dict = {}

    with open(fSTAR, 'r') as f:
        for num, line in enumerate(f):
            sline = line.strip()
            if not sline or (' ' not in sline):
                continue
            match = re.match(r'_rln(\w+)\s+#(\d+)', sline)
            if match:
                header_dict[match.group(1).lower()] = int(match.group(2)) - 1
                continue
            sp = sline.split()
            try:
                volt = float(sp[header_dict['voltage']]) * 1000.0
                dU = float(sp[header_dict['defocusu']])
                dV = float(sp[header_dict['defocusv']])
                dT = float(sp[header_dict['defocusangle']]) * (math.pi / 180)
                ps = float(sp[header_dict['phaseshift']]) * (math.pi / 180)
                c = float(sp[header_dict['sphericalaberration']]) * 1e7
                aC = float(sp[header_dict['amplitudecontrast']])
                iN = sp[header_dict['imagename']]
                mN = sp[header_dict['micrographname']]
                coordX = float(sp[header_dict['coordinatex']])
                coordY = float(sp[header_dict['coordinatey']])
            except (ValueError, IndexError):
                sys.stderr.write(
                    'Warning: skipping line #{} ({}) that cannot be parsed\n'.format(num + 1, sline))
                continue
            except KeyError as e:
                sys.stderr.write(
                    'Header does not include {}. This is an invalid star file\n'.format(str(e)))
                sys.exit(2)

            print '{volt:18.6f} \
                   {dU:18.6f} \
                   {dV:18.6f} \
                   {dT:18.6f} \
                   {c:18.6f} \
                   {aC:18.6f} \
                   {phaseShift:18.6f} \
                   {iN} \
                   {mN} \
                   {coordX:18.6f} \
                   {coordY:18.6f} \
                   {gI} \
                   {classI} \
                   {quat0:18.6f} \
                   {quat1:18.6f} \
                   {quat2:18.6f} \
                   {quat3:18.6f} \
                   {stdRot:18.6f} \
                   {transX:18.6f} \
                   {transY:18.6f} \
                   {stdTransX:18.6f} \
                   {stdTransY:18.6f} \
                   {df:18.6f} \
                   {stdDf:18.6f} \
                   {score:18.6f}'.format(volt=volt,
                                         dU=dU,
                                         dV=dV,
                                         dT=dT,
                                         c=c,
                                         aC=aC,
                                         phaseShift=ps,
                                         iN=iN,
                                         mN=mN,
                                         coordX=coordX,
                                         coordY=coordY,
                                         gI=1,
                                         classI=0,
                                         quat0=0,
                                         quat1=0,
                                         quat2=0,
                                         quat3=0,
                                         stdRot=0,
                                         transX=0,
                                         transY=0,
                                         stdTransX=0,
                                         stdTransY=0,
                                         df=1,
                                         stdDf=0,
                                         score=0)

main()
