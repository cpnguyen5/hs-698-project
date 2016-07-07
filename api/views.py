from api import app
from flask import render_template, url_for, jsonify
import os
import project
import sqlite3
import numpy as np
import pandas as pd
import json
from .models import Report
from api import db
from sqlalchemy import func


def get_abs_path():
    """
    This function takes no parameters and returns the api root directory pathway.
    :return: api directory pathway
    """
    return os.path.abspath(os.path.dirname(__file__))


def get_db():
    db_path=os.path.join(project.get_path(),'cms3.db')
    if not os.path.isfile(db_path):
        # project.create_table() #sqlite3
        project.init_db() #sqlalchemy
    return db_path


@app.route('/')
def home():
    return "HS 698 Flask Web Development Project"

@app.route('/state')
def state():

    rows = db.session.query(Report.provider_state_code, func.avg(Report.percent_of_beneficiaries_identified_with_cancer)).\
        order_by(Report.provider_state_code).group_by(Report.provider_state_code).all()
    state = rows

    # db_file = get_db()
    # conn = sqlite3.connect(db_file)  # create/open database
    # conn.row_factory=sqlite3.Row
    #
    # with conn:
    #     c = conn.cursor()
    #
    #     c.execute('''SELECT provider_state_code, percent_of_beneficiaries_identified_with_cancer
    #     FROM report
    #
    #     ORDER BY provider_state_code;''')
    #     # GROUP BY provider_state_code
    #     rows = c.fetchall()
    #
    # states_dict={}
    # for row in rows:
    #     # print row["provider_state_code"]
    #     key = str(row[0])
    #     if row[1] != None and isinstance(row[1], int):
    #         if states_dict.has_key(key):
    #             states_dict[key]+=[row[1]]
    #         else:
    #             states_dict[str(key)]=[row[1]]
    #     else:
    #         pass
    #
    # # for elem in states_dict.items():
    # #     print elem
    # dict = sorted(states_dict.items())
    # for i, value in enumerate(dict):
    #     temp=list(dict[i])
    #     temp[1]=(sum(dict[i][1])) / (len(dict[i][1]))
    #     dict[i]=tuple(temp)
    #     # print dict[i]
    # # for i in range(len(dict)):
    # #     print dict[i][0], sum(dict[i][1])
    #     # dict[i][1]=sum(dict[i][1])
    return render_template("state.html", rows=state)


@app.route('/map')
def map():

    rows = db.session.query(Report.provider_state_code, func.avg(Report.percent_of_beneficiaries_identified_with_cancer)).\
        filter(Report.provider_state_code != 'DC').order_by(Report.provider_state_code).group_by(Report.provider_state_code).all()

    state_lst=[]
    for i in range(len(rows)):
        state = tuple()
        state += (rows[i][0],)
        state += (round(rows[i][1],2),)
        state_lst+=[state]
    dict_state = dict(state_lst)

    # db_file = get_db()
    # conn = sqlite3.connect(db_file)  # create/open database
    # conn.row_factory = sqlite3.Row
    # #
    # with conn:
    #     c = conn.cursor()
    #     #
    #     c.execute('''SELECT provider_state_code, percent_of_beneficiaries_identified_with_cancer
    #      FROM report
    #
    #      ORDER BY provider_state_code;''')
    #     # GROUP BY provider_state_code
    #     rows = c.fetchall()

    # #Create dictionary with state-value pairs -- values are list of values
    # states_dict = {}
    # for row in rows:
    #     # print row["provider_state_code"]
    #     key = str(row[0])
    #     if row[1] != None and isinstance(row[1], int):
    #         if states_dict.has_key(key):
    #             states_dict[key] += [row[1]]
    #         else:
    #             states_dict[str(key)] = [row[1]]
    #     else:
    #         pass

    # #Obtain list of tuples with key-value pairs of state-average prevalence
    # lst_pair = sorted(states_dict.items())
    # for i, value in enumerate(lst_pair):
    #     temp = list(lst_pair[i])
    #     temp[1] = (sum(lst_pair[i][1])) / (len(lst_pair[i][1]))
    #     lst_pair[i] = tuple(temp)
    #
    # state = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA',
    #          'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK',
    #          'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']
    # #filter list for actual states
    # lst_state = []
    # for elem in lst_pair:
    #     if str(elem[0]) in state:
    #         lst_state += [elem]
    # dict_state=dict(lst_state)
    return render_template("map.html", d_state=dict_state, js_file=url_for('static',
                                                                           filename='js/datamaps.usa.min.js'))


@app.route('/cost')
def cost():

    rows = db.session.query(Report.provider_state_code, func.avg(Report.total_medicare_standardized_payment_amount)).\
        filter(Report.provider_state_code != 'DC').order_by(Report.provider_state_code).group_by(Report.provider_state_code).all()

    state_lst=[]
    for i in range(len(rows)):
        state = tuple()
        state += (rows[i][0],)
        state += (round(rows[i][1],2),)
        state_lst+=[state]
    state_cost=pd.DataFrame(state_lst, dtype=int)
    csv_path = os.path.join(get_abs_path(), 'static', 'tmp', 'state_cost.csv')
    state_cost.to_csv(csv_path, index=False, header= ["name","value"])
    return render_template("state_cost.html", data_file = url_for('static',
                                                                  filename='tmp/state_cost.csv'))

@app.route('/cost/top')
def top_cost():

    rows = db.session.query(Report.provider_state_code, func.avg(Report.total_medicare_standardized_payment_amount),
                            func.avg(Report.number_of_beneficiaries_age_less_65),
                            func.avg(Report.number_of_beneficiaries_age_65_to_74),
                            func.avg(Report.number_of_beneficiaries_age_75_to_84),
                            func.avg(Report.number_of_beneficiaries_age_greater_84)).\
        filter(Report.provider_state_code != 'DC').order_by(func.avg(Report.total_medicare_standardized_payment_amount).desc()).\
        group_by(Report.provider_state_code).limit(5).all()

    data = []
    for row in rows:
        state_sum=np.sum(row[2:])
        state_cost=tuple()
        state_cost+=(row[0],) #state
        print row[0]
        state_cost+=(round(row[1], 2),) #total payment amount
        state_cost+=( int(((float(row[2])) / state_sum) * row[1]),) #<65
        state_cost+=( int(((float(row[3])) / state_sum) * row[1]),) #65 to 74
        state_cost+=( int(((float(row[4])) / state_sum) * row[1]),) #75 to 84
        state_cost+=( int(((float(row[5])) / state_sum) * row[1]),) #>85
        data+=[state_cost]
    return render_template("cost.html", data=data)


@app.route('/data')
def data():

    db_file = get_db()
    conn = sqlite3.connect(db_file)  # create/open database
    conn.row_factory=sqlite3.Row

    # with conn:
    #     c = conn.cursor()
    #
    #     c.execute('''SELECT * FROM report''')
    #     rows = c.fetchall()

    columns = ["npi", "provider_last_name", "provider_first_name", "provider_middle_initial", "provider_credentials",
               "provider_gender", "provider_entity_type", "provider_street_address_1", "provider_street_address_2",
               "provider_city",
               "provider_zip_code", "provider_state_code", "provider_country_code", "provider_type",
               "medicare_participation_indicator",
               "number_of_HCPCS", "number_of_services", "number_of_medicare_beneficiaries",
               "total_submitted_charge_amount",
               "total_medicare_allowed_amount", "total_medicare_payment_amount",
               "total_medicare_standardized_payment_amount",
               "drug_suppress_indicator", "number_of_HCPCS_associated_with_drug_services", "number_of_drug_services",
               "number_of_medicare_beneficiaries_with_drug_services", "total_drug_submitted_charge_amount",
               "total_drug_medicare_allowed_amount",
               "total_drug_medicare_payment_amount", "total_drug_medicare_standardized_payment_amount",
               "medical_suppress_indicator",
               "number_of_HCPCS_associated_medical_services", "number_of_medical_services",
               "number_of_medicare_beneficiaries_with_medical_services",
               "total_medical_submitted_charge_amount", "total_medical_medicare_allowed_amount",
               "total_medical_medicare_payment_amount",
               "total_medical_medicare_standardized_payment_amount", "average_age_of_beneficiaries",
               "number_of_beneficiaries_age_less_65",
               "number_of_beneficiaries_age_65_to_74", "number_of_beneficiaries_age_75_to_84",
               "number_of_beneficiaries_age_greater_84",
               "number_of_female_beneficiaries", "number_of_male_beneficiaries",
               "number_of_non_hispanic_white_beneficiaries",
               "number_of_african_american_beneficiaries", "number_of_asian_pacific_islander_beneficiaries",
               "number_of_hispanic_beneficiaries",
               "number_of_american_indian_alaskan_native_beneficiaries",
               "number_of_beneficiaries_with_race_not_elsewhere_classified",
               "number_of_beneficiaries_with_medicare_only_entitlement",
               "number_of_beneficiaries_with_medicare_and_medicaid_entitlement",
               "percent_of_beneficiaries_identified_with_atrial_fibrillation",
               "percent_of_beneficiaries_identified_with_alzheimers_disease_or_dementia",
               "percent_of_beneficiaries_identified_with_asthma", "percent_of_beneficiaries_identified_with_cancer",
               "percent_of_beneficiaries_identified_with_heart_failure",
               "percent_of_beneficiaries_identified_with_chronic_kidney_disease",
               "percent_of_beneficiaries_identified_with_chronic_obstructive_pulmonary_disease",
               "percent_of_beneficiaries_identified_with_depression",
               "percent_of_beneficiaries_identified_with_diabetes",
               "percent_of_beneficiaries_identified_with_hyperlipidemia",
               "percent_of_beneficiaries_identified_with_hypertension",
               "percent_of_beneficiaries_identified_with_ischemic_heart_disease",
               "percent_of_beneficiaries_identified_with_osteoporosis",
               "percent_of_beneficiaries_identified_with_rheumatoid_arthritis_osteoarthritis",
               "percent_of_beneficiaries_identified_with_schizophrenia_other_psychotic_disorders",
               "percent_of_beneficiaries_identified_with_stroke",
               "average_HCC_risk_score_of_beneficiaries"]

    df=pd.read_sql('SELECT * FROM report;', conn, columns=columns)
    data = df.head().as_matrix().astype(str)
    return render_template("data.html", data=data)
    # df = df.head() #head - dataframe
    # data = json.loads(df.to_json()) #exports data frame as json string --> load/parsed json into python object (dict or lsit)
    # return jsonify(data)


##main
