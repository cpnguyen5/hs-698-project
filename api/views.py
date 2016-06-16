from api import app
from flask import render_template, url_for, jsonify
import os
import project
import sqlite3


def get_data():

    project.query()

@app.route('/')
def home():
    return "Flask Web Development Project"

@app.route('/state')
def state():

    path = project.get_path()
    db_file = os.path.join(path, 'cms1.db')
    conn = sqlite3.connect(db_file)  # create/open database
    conn.row_factory=sqlite3.Row

    with conn:
        c = conn.cursor()

        c.execute('''SELECT provider_state_code, percent_of_beneficiaries_identified_with_cancer
        FROM report

        ORDER BY provider_state_code;''')
        # GROUP BY provider_state_code
        rows = c.fetchall()

    states_dict={}
    for row in rows:
        # print row["provider_state_code"]
        key = str(row[0])
        if row[1] != None and isinstance(row[1], int):
            if states_dict.has_key(key):
                states_dict[key]+=[row[1]]
            else:
                states_dict[str(key)]=[row[1]]
        else:
            pass

    # for elem in states_dict.items():
    #     print elem
    dict = sorted(states_dict.items())
    for i, value in enumerate(dict):
        temp=list(dict[i])
        temp[1]=(sum(dict[i][1])) / (len(dict[i][1]))
        dict[i]=tuple(temp)
        # print dict[i]
    # for i in range(len(dict)):
    #     print dict[i][0], sum(dict[i][1])
        # dict[i][1]=sum(dict[i][1])
    return render_template("state.html", rows=dict)
