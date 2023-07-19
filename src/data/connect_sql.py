'''
import csv
import MySQLdb
mydb = MySQLdb.connect(host = '127.0.0.1', user = 'root', password = "", database = 'all_db')
path = '/home/bhatti/dataset/VCHAMPS/procedures_train.csv'

with open(path) as csv_file:
    csvfile = csv.reader(csv_file, delimiter = ',')
    all_value = []
    for row in csv_file:
        value = ()
'''

import mysql.connector as msql
from mysql.connector import Error

try:
    conn = msql.connect(host='127.0.0.1', user='root',
                        password='Dc1204+-*', port = '3306')
    if conn.is_connected():
        cursor = conn.cursor()
        cursor.execute("CREATE DATABASE ProcedureDB")
        print("ProcedureDB database is created")

except Error as e:
    print("Error while connecting to MySQL", e)