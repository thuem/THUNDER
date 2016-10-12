import sqlite3
import math

fSTAR = "main.star"

fDB = "C15.db"

iVoltage = 0

iDefocusU = 1

iDefocusV = 2

iDefocusTheta = 3

iCs = 4

iImageName = 9

iMacrographName = 13

lines = open(fSTAR, "r").readlines()

print "There is ", len(lines), " particles in total."

voltage = [float(line.strip().split()[iVoltage]) * 1000 for line in lines]

defocusU = [float(line.strip().split()[iDefocusU]) for line in lines]

defocusV = [float(line.strip().split()[iDefocusV]) for line in lines]

defocusTheta = [float(line.strip().split()[iDefocusTheta]) / 180 * math.pi for line in lines]

CS = [float(line.strip().split()[iCs]) for line in lines]

imageName = [line.strip().split()[iImageName] for line in lines]

micrographName = [line.strip().split()[iMacrographName] for line in lines]

conn = sqlite3.connect(fDB)

print "Open Database Successfully"

conn.execute('''CREATE TABLE micrographs
                (ID             INT     PRIMARY KEY,
                 Name           TEXT,
                 Voltage        REAL                  NOT NULL,
                 DefocusU       REAL                  NOT NULL,
                 DefocusV       REAL                  NOT NULL,
                 DefocusAngle   REAL                  NOT NULL,
                 Cs             REAL                  NOT NULL);''')

conn.execute('''CREATE TABLE particles
                (ID             INT     PRIMARY KEY,
                 Name           TEXT,
                 GroupID        INT                   NOT NULL,
                 MicrographID   INT                   NOT NULL);''')

conn.execute('''CREATE TABLE groups
                (ID             INT     PRIMARY KEY,
                 Name           TEXT);''')

conn.commit()

print "Tables Created"

micrographList = []

for i in range(len(lines)):

    micrographID = 0
    groupID = 0

    if micrographName[i] not in micrographList:

        micrographID = len(micrographList) + 1

        micrographList.append(micrographName[i])

        sql = "INSERT INTO micrographs (ID, Name, Voltage, DefocusU, \
               DefocusV, DefocusAngle, Cs) VALUES (" \
            + str(micrographID) + ", '" \
            + micrographName[i] + "', " \
            + str(voltage[i]) + ", " \
            + str(defocusU[i]) + ", " \
            + str(defocusV[i]) + ", " \
            + str(defocusTheta[i]) + ", " \
            + str(CS[i]) + ");"

        conn.execute(sql)

        conn.commit()

        sql = "INSERT INTO groups (ID, Name) VALUES (" \
            + str(micrographID) \
            + ", '');"

        conn.execute(sql)

        conn.commit()

        # groupID = micrographID

        # sql = "INSERT INTO groups (ID, Name) VALUES (" \
          #   + str(groupID) + ", '')"
        # conn.execute(sql)

    else:

        micrographID = micrographList.index(micrographName[i]) + 1

        # groupID = micrographID

    #groupID = 1

    sql = "INSERT INTO particles (ID, Name, GroupID, MicrographID) VALUES (" \
        + str(i + 1) + ", '" \
        + imageName[i] + "', " \
        + str(micrographID) + ", " \
        + str(micrographID) +");";

    conn.execute(sql)

    conn.commit()

#print micrographList

print "There is ", len(micrographList), "micrographs in total."
