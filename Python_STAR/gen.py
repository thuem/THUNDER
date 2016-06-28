import sqlite3

lines = open("main.star", "r").readlines()
# print lines

print "There is ", len(lines), " particles in total."

voltage = [float(line.strip().split()[0]) for line in lines]
# print voltage

defocusU = [float(line.strip().split()[1]) for line in lines]
# print defocusU

defocusV = [float(line.strip().split()[2]) for line in lines]
# print defocusU

defocusTheta = [float(line.strip().split()[3]) for line in lines]
# print defocusTheta

imageName = [line.strip().split()[9] for line in lines]
# print imageName

micrographName = [line.strip().split()[13] for line in lines]
# print micrographName

conn = sqlite3.connect("C15.db")

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
    micrographID = 1
    if micrographName[i] not in micrographList:
        micrographList.append(micrographName[i])
        sql = "INSERT INTO micrographs (Name, Voltage, DefocusU, \
               DefocusV, DefocusAngle, Cs) VALUES ('" \
            + micrographName[i] + "', " \
            + str(voltage[i]) + ", " \
            + str(defocusU[i]) + ", " \
            + str(defocusV[i]) + ", " \
            + str(defocusTheta[i]) + ", " + "0);"
        conn.execute(sql)
        conn.commit()
        micrographID = len(micrographList)
    else:
        micrographID = micrographList.index(micrographName[i]) + 1

    groupID = micrographID

    sql = "INSERT INTO particles (Name, GroupID, MicrographID) VALUES ('" \
        + imageName[i] + "', " \
        + str(groupID) + ", " \
        + str(micrographID) +");";
    conn.execute(sql)
    conn.commit()

#print micrographList

print "There is ", len(micrographList), "micrographs in total."
