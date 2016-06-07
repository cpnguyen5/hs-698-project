import sqlite3
import os

def get_path():
    f_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'dataset', 'cms.db')
    return f_name


def create_table(path):
    return


def query(path):

    head_path, file = os.path.split(path)
    queries_path = os.path.join(head_path, "cms_queries.sql")
    print queries_path
    conn = sqlite3.connect(path)
    print "Connected to database successfully"

    c = conn.cursor()

    print "Reading SQL script..."
    sql_file = open(queries_path, 'r')
    sql = sql_file.read()
    sql_file.close()

    print "Running SQL script..."
    # c.executescript(sql)
    c = conn.execute("SELECT provider_state_code, percent_of_beneficiaries_identified_with_cancer FROM report GROUP BY provider_state_code;")
    # for row in c:
    #     print row
    print tuple(c)[1][1]

    # all_rows = q.fetchone()
    # print all_rows
    return


query(get_path())

conn = sqlite3.connect(get_path())
c = conn.cursor()

# c.execute('SELECT npi from puf WHERE npi < 1300000000')
# print c.fetchone()
# print c.fetchone()
#
# print "\n", c.fetchall()