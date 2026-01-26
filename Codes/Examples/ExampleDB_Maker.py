#!/usr/bin/python3.11
# -*- coding: utf-8 -*-

import sqlite3;
import pandas as pd;

PWT_DB = 'PWT2023.db';
EXPORT_DB = 'HS2_Export2024.db';


meow = pd.read_stata('data/pwt110.dta');
meow = meow[meow['year'] == 2023];
conn = sqlite3.connect('data/' + PWT_DB);
meow.to_sql('pwt2023', conn, if_exists='replace', index=False);
conn.close();


with open('data/BACI_HS96_2d_2024_v202601.csv', 'r') as f:
    ExportData = [x.strip().split(',')  for x in f.readlines()[1:]];

ExportData =[(int(x[0]),int(x[1]), float(x[2]) ) for x in  ExportData ];


with open('data/country_codes_V202601.csv', 'r') as f:
    Iso3Info = [x.strip().split(',')  for x in f.readlines()[1:]]

Iso3Info = [(int(x[0]), x[3], x[1]) for  x in Iso3Info];


with open('data/HS2_1996.txt', 'r') as f:
    HS2_Names = [x.strip().split(':')  for x in f.readlines()]

HS2_Names = [(int(x[0]), x[1]) for  x in HS2_Names];


conn = sqlite3.connect('data/' + EXPORT_DB);
cursor = conn.cursor();

cursor.execute('''
    CREATE TABLE Export2024 (
    origin_id INTEGER NOT NULL, 
    hs2 INTEGER NOT NULL,
    exp_value REAL NOT NULL
    )
    ''');

cursor.executemany('''INSERT INTO Export2024 
    (origin_id, hs2, exp_value) 
    VALUES (?,?,?)''', ExportData);


cursor.execute('''
    CREATE TABLE Iso3Info (
    origin_id INTEGER NOT NULL PRIMARY KEY,
    iso3 TEXT(3) NOT NULL,
    name TEXT NOT NULL
    )
    ''');

cursor.executemany('''INSERT INTO Iso3Info 
    (origin_id, iso3, name) 
    VALUES (?,?,?)''', Iso3Info);


cursor.execute('''
    CREATE TABLE HsInfo (
    hs2 INTEGER NOT NULL PRIMARY KEY,
    name TEXT NOT NULL
    )
    ''');

cursor.executemany('''INSERT INTO HsInfo 
    (hs2, name) 
    VALUES (?,?)''', HS2_Names);



conn.commit()
conn.execute('VACUUM;');
cursor.close()
conn.close()
