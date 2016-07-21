from api import app
from flask import render_template, url_for, jsonify
import os
import project
import sqlite3
import numpy as np
import pandas as pd
import json
from .models import Report, Puf, Cancer
from api import db
from sqlalchemy import func
from sqlalchemy import select
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import decomposition
import matplotlib.pyplot as plt
import seaborn as sns

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
    return render_template("home.html", img_file=url_for('static', filename='img/cms_logo.jpg'))




@app.route('/prevalence')
def prevalence():
    rows = db.session.query(Report.provider_state_code,func.avg(Report.percent_of_beneficiaries_identified_with_cancer),
                            func.avg(Report.percent_of_beneficiaries_identified_with_atrial_fibrillation),
                            func.avg(Report.percent_of_beneficiaries_identified_with_alzheimers_disease_or_dementia),
                            func.avg(Report.percent_of_beneficiaries_identified_with_asthma),
                            func.avg(Report.percent_of_beneficiaries_identified_with_heart_failure),
                            func.avg(Report.percent_of_beneficiaries_identified_with_chronic_kidney_disease),
                            func.avg(Report.percent_of_beneficiaries_identified_with_chronic_obstructive_pulmonary_disease),
                            func.avg(Report.percent_of_beneficiaries_identified_with_depression),
                            func.avg(Report.percent_of_beneficiaries_identified_with_diabetes),
                            func.avg(Report.percent_of_beneficiaries_identified_with_hyperlipidemia),
                            func.avg(Report.percent_of_beneficiaries_identified_with_hypertension),
                            func.avg(Report.percent_of_beneficiaries_identified_with_ischemic_heart_disease),
                            func.avg(Report.percent_of_beneficiaries_identified_with_osteoporosis),
                            func.avg(Report.percent_of_beneficiaries_identified_with_rheumatoid_arthritis_osteoarthritis),
                            func.avg(Report.percent_of_beneficiaries_identified_with_schizophrenia_other_psychotic_disorders),
                            func.avg(Report.percent_of_beneficiaries_identified_with_stroke)).\
        order_by(Report.provider_state_code).group_by(Report.provider_state_code).all()

    overall_prev = db.session.query(func.avg(Report.percent_of_beneficiaries_identified_with_cancer),
                                    func.avg(Report.percent_of_beneficiaries_identified_with_atrial_fibrillation),
                                    func.avg(
                                        Report.percent_of_beneficiaries_identified_with_alzheimers_disease_or_dementia),
                                    func.avg(Report.percent_of_beneficiaries_identified_with_asthma),
                                    func.avg(Report.percent_of_beneficiaries_identified_with_heart_failure),
                                    func.avg(Report.percent_of_beneficiaries_identified_with_chronic_kidney_disease),
                                    func.avg(
                                        Report.percent_of_beneficiaries_identified_with_chronic_obstructive_pulmonary_disease),
                                    func.avg(Report.percent_of_beneficiaries_identified_with_depression),
                                    func.avg(Report.percent_of_beneficiaries_identified_with_diabetes),
                                    func.avg(Report.percent_of_beneficiaries_identified_with_hyperlipidemia),
                                    func.avg(Report.percent_of_beneficiaries_identified_with_hypertension),
                                    func.avg(Report.percent_of_beneficiaries_identified_with_ischemic_heart_disease),
                                    func.avg(Report.percent_of_beneficiaries_identified_with_osteoporosis),
                                    func.avg(
                                        Report.percent_of_beneficiaries_identified_with_rheumatoid_arthritis_osteoarthritis),
                                    func.avg(
                                        Report.percent_of_beneficiaries_identified_with_schizophrenia_other_psychotic_disorders),
                                    func.avg(Report.percent_of_beneficiaries_identified_with_stroke)).all()[0]
    state_avg = []
    for elem in rows:
        state_tup = tuple()
        state_tup += (elem[0],)
        i=1
        while i < len(elem):
            state_tup += (round(elem[i],2),)
            i+=1
        state_avg += [state_tup]
    overall_round = []
    for i in range(len(overall_prev)):
        overall_round += [round(float(overall_prev[i]), 2)]
    state_avg += [('Total Avg',) + tuple(overall_round)]
    diseases = ["Cancer", "A-Fib", "Alzheimers", "Asthma", "Heart Fail",
                "Kidney Dis", "Pulmonary Dis", "Depression", "Diabetes",
                "Hyperlipidemia", "Hypertension", "Ischemic Heart Dis", "Osteoporosis",\
                "Rheumatoid Arthritis", "Schizophrenia", "Stroke"]
    overall_freq = []
    for i in range(len(overall_round)):
        overall_freq += [round((overall_round[i] / 100), 5)]
    overall_bar = pd.DataFrame(np.column_stack((diseases, overall_freq)))
    tsv_path = os.path.join(get_abs_path(), 'static', 'tmp', 'overall_prev.tsv')
    overall_bar.to_csv(tsv_path, sep='\t', header=['disease', 'frequency'])
    return render_template("state.html", rows=state_avg, prev_file=url_for('static', filename='tmp/overall_prev.tsv'))


@app.route('/prevalence/top')
def top_prev():
    rows = db.session.query(Report.percent_of_beneficiaries_identified_with_hypertension,
                            Report.percent_of_beneficiaries_identified_with_hyperlipidemia,
                            Report.percent_of_beneficiaries_identified_with_rheumatoid_arthritis_osteoarthritis,
                            Report.percent_of_beneficiaries_identified_with_ischemic_heart_disease,
                            Report.percent_of_beneficiaries_identified_with_diabetes,
                            Report.number_of_beneficiaries_age_less_65,
                            Report.number_of_beneficiaries_age_65_to_74,
                            Report.number_of_beneficiaries_age_75_to_84,
                            Report.number_of_beneficiaries_age_greater_84).all()
    col=['perc_hypertension', 'perc_hyperlipidemia', 'perc_arthritis', 'perc_ischemic_heart', 'perc_diabetes',
         'num_0-64', 'num_65-74', 'num_75-84', 'num_85']
    prev_df = pd.DataFrame(np.array(rows), columns=col, dtype=np.float64)
    corr = prev_df.corr()
    print corr.iloc[:5]
    return "Hi"

@app.route('/cancer')
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
    return render_template("map.html", d_state=dict_state, rows=state_lst, js_file=url_for('static',
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

    rows_age = db.session.query(func.sum(Report.total_medicare_standardized_payment_amount),
                                func.sum(Report.total_medical_medicare_standardized_payment_amount),
                                func.sum(Report.total_drug_medicare_standardized_payment_amount),
                                func.sum(Report.number_of_beneficiaries_age_less_65),
                                func.sum(Report.number_of_beneficiaries_age_65_to_74),
                                func.sum(Report.number_of_beneficiaries_age_75_to_84),
                                func.sum(Report.number_of_beneficiaries_age_greater_84)).all()
    rows_age = list(rows_age[0])
    total_age = sum(rows_age[3:])
    age_0_64 = float(rows_age[3]) / total_age
    age_65_74 = float(rows_age[4]) / total_age
    age_75_84 = float(rows_age[5]) / total_age
    age_85 = float(rows_age[6]) / total_age
    age = [age_0_64, age_65_74, age_75_84, age_85]
    medicare_age_amt = []
    medicare_medical_amt = []
    medicare_drug_amt = []
    for i in range(len(age)):
        medicare_age_amt += [rows_age[0] * age[i]]
        medicare_medical_amt += [rows_age[1] * age[i]]
        medicare_drug_amt += [rows_age[2] * age[i]]
    costs = ['Medicare Amount', 'Medical Amount', 'Drug Amount']
    age_data = np.vstack((medicare_age_amt, medicare_medical_amt, medicare_drug_amt))
    age_data = np.column_stack((costs, age_data))
    col = ['costs', 'age_0-64', 'age_65-74', 'age_75-84', 'age_84']
    age_df = pd.DataFrame(age_data, columns=col)
    age_path = os.path.join(get_abs_path(), 'static', 'tmp', 'cost_age.csv')
    age_df.to_csv(age_path, sep=',', header=col)


    top_rows = db.session.query(Report.provider_state_code, func.avg(Report.total_medicare_standardized_payment_amount),
                            func.avg(Report.number_of_beneficiaries_age_less_65),
                            func.avg(Report.number_of_beneficiaries_age_65_to_74),
                            func.avg(Report.number_of_beneficiaries_age_75_to_84),
                            func.avg(Report.number_of_beneficiaries_age_greater_84)). \
        filter(Report.provider_state_code != 'DC').order_by(
        func.avg(Report.total_medicare_standardized_payment_amount).desc()). \
        group_by(Report.provider_state_code).limit(5).all()
    data = []
    for row in top_rows:
        state_sum=np.sum(row[2:])
        state_cost=tuple()
        state_cost+=(row[0],) #state
        state_cost+=(round(row[1], 2),) #total payment amount
        state_cost+=( int(((float(row[2])) / state_sum) * row[1]),) #<65
        state_cost+=( int(((float(row[3])) / state_sum) * row[1]),) #65 to 74
        state_cost+=( int(((float(row[4])) / state_sum) * row[1]),) #75 to 84
        state_cost+=( int(((float(row[5])) / state_sum) * row[1]),) #>85
        data+=[state_cost]
    return render_template("state_cost.html",
                           data_file = url_for('static', filename='tmp/state_cost.csv'), data=data)



@app.route('/cost/demo')
def demographics():
    ##age
    rows_age = db.session.query(func.sum(Report.total_medicare_standardized_payment_amount),
                                func.sum(Report.total_medical_medicare_standardized_payment_amount),
                                func.sum(Report.total_drug_medicare_standardized_payment_amount),
                                func.sum(Report.number_of_beneficiaries_age_less_65),
                                func.sum(Report.number_of_beneficiaries_age_65_to_74),
                                func.sum(Report.number_of_beneficiaries_age_75_to_84),
                                func.sum(Report.number_of_beneficiaries_age_greater_84)).all()
    rows_age = list(rows_age[0])
    total_age = sum(rows_age[3:])
    age_0_64 = float(rows_age[3]) / total_age
    age_65_74 = float(rows_age[4]) / total_age
    age_75_84 = float(rows_age[5]) / total_age
    age_85 = float(rows_age[6]) / total_age
    age = [age_0_64, age_65_74, age_75_84, age_85]
    medicare_amt_age = []
    medicare_medical_amt_age = []
    medicare_drug_amt_age = []
    for i in range(len(age)):
        medicare_amt_age += [round(rows_age[0] * age[i],2)]
        medicare_medical_amt_age += [round(rows_age[1] * age[i],2)]
        medicare_drug_amt_age += [round(rows_age[2] * age[i],2)]
    costs = ['Medicare Amount ($)', 'Medical Amount ($)', 'Drug Amount ($)']
    age_data = np.vstack((medicare_amt_age, medicare_medical_amt_age, medicare_drug_amt_age))
    age_df = pd.DataFrame({'costs': costs,
                           'age 0-64': age_data[:,0],
                           'age 65-74': age_data[:,1],
                           'age 75-84': age_data[:,2],
                           'age 84+': age_data[:,3]})
    age_path = os.path.join(get_abs_path(), 'static', 'tmp', 'cost_age.csv')
    age_df.to_csv(age_path, sep=',', index=False)
    costs_age = np.column_stack((costs+['Total'], np.vstack((age_data, np.sum(age_data, axis=0)))))
    age_perc = np.round(age, 4) * 100
    age_ratio = np.column_stack((['Count (n)', 'Percentage(%)'],np.vstack((rows_age[3:], age_perc))))


    ##race
    rows_race = db.session.query(func.sum(Report.number_of_non_hispanic_white_beneficiaries),
                                 func.sum(Report.number_of_african_american_beneficiaries),
                                 func.sum(Report.number_of_asian_pacific_islander_beneficiaries),
                                 func.sum(Report.number_of_hispanic_beneficiaries),
                                 func.sum(Report.number_of_american_indian_alaskan_native_beneficiaries),
                                 func.sum(Report.number_of_beneficiaries_with_race_not_elsewhere_classified)).all()
    rows_race = rows_race[0]
    total_race = sum(rows_race)
    white = float(rows_race[0]) / total_race
    african_am = float(rows_race[1]) / total_race
    api = float(rows_race[2]) / total_race
    hispanic = float(rows_race[3]) / total_race
    native_am = float(rows_race[4]) / total_race
    other_race = float(rows_race[5]) / total_race
    race = [white, african_am, api, hispanic, native_am, other_race]
    medicare_amt_race = []
    medicare_medical_amt_race = []
    medicare_drug_amt_race = []
    for i in range(len(race)):
        medicare_amt_race += [round(rows_age[0] * race[i],2)]
        medicare_medical_amt_race += [round(rows_age[1] * race[i],2)]
        medicare_drug_amt_race += [round(rows_age[2] * race[i],2)]
    race_data = np.vstack((medicare_amt_race, medicare_medical_amt_race, medicare_drug_amt_race))
    race_df = pd.DataFrame({'costs': costs,
                            'White': race_data[:, 0],
                            'African-American': race_data[:, 1],
                            'Asian-Pacific Islander': race_data[:, 2],
                            'Hispanic': race_data[:, 3],
                            'Native American': race_data[:, 4],
                            'Other Race': race_data[:, 5]})
    race_path = os.path.join(get_abs_path(), 'static', 'tmp', 'cost_race.csv')
    race_df.to_csv(race_path, sep=',', index=False)
    costs_race = np.column_stack((costs+['Total'], np.vstack((race_data, np.sum(race_data, axis=0)))))
    race_perc = np.round(race, 4) * 100
    race_ratio = np.column_stack((['Count (n)', 'Percentage(%)'],np.vstack((rows_race[:6], race_perc))))

    ##sex
    rows_sex = db.session.query(func.sum(Report.number_of_female_beneficiaries),
                                 func.sum(Report.number_of_male_beneficiaries)).all()

    rows_sex = rows_sex[0]
    total_sex = sum(rows_sex)
    female = float(rows_sex[0]) / total_sex
    male = float(rows_sex[1]) / total_sex
    sex = [female, male]
    medicare_amt_sex = []
    medicare_medical_amt_sex = []
    medicare_drug_amt_sex = []
    for i in range(len(sex)):
        medicare_amt_sex += [round(rows_age[0] * sex[i], 2)]
        medicare_medical_amt_sex += [round(rows_age[1] * sex[i], 2)]
        medicare_drug_amt_sex += [round(rows_age[2] * sex[i], 2)]
    sex_data = np.vstack((medicare_amt_sex, medicare_medical_amt_sex, medicare_drug_amt_sex))
    sex_df = pd.DataFrame({'costs': costs,
                           'Female': sex_data[:, 0],
                           'Male': sex_data[:, 1]})
    costs_sex = np.column_stack((costs+['Total'], np.vstack((sex_data, np.sum(sex_data, axis=0)))))
    sex_perc = np.round(sex, 4) * 100
    sex_ratio = np.column_stack((['Count (n)', 'Percentage (%)'],np.vstack((rows_sex[:2], sex_perc))))
    sex_path = os.path.join(get_abs_path(), 'static', 'tmp', 'cost_sex.csv')
    sex_df.to_csv(sex_path, sep=',', index=False)

    ##heatmap
    rows_heatmap = db.session.query(Report.total_medicare_standardized_payment_amount,
                                    Report.total_medical_medicare_standardized_payment_amount,
                                    Report.total_drug_medicare_standardized_payment_amount,
                                    Report.number_of_beneficiaries_age_less_65,
                                    Report.number_of_beneficiaries_age_65_to_74,
                                    Report.number_of_beneficiaries_age_75_to_84,
                                    Report.number_of_beneficiaries_age_greater_84,
                                    Report.number_of_non_hispanic_white_beneficiaries,
                                    Report.number_of_african_american_beneficiaries,
                                    Report.number_of_asian_pacific_islander_beneficiaries,
                                    Report.number_of_hispanic_beneficiaries,
                                    Report.number_of_american_indian_alaskan_native_beneficiaries,
                                    Report.number_of_beneficiaries_with_race_not_elsewhere_classified,
                                    Report.number_of_female_beneficiaries, Report.number_of_male_beneficiaries).all()
    col = ['medicare_amount', 'medicare_medical_amount', 'medicare_drug_amount', 'num_age_less_65', 'num_age_65_to_74',
           'num_age_75-84', 'num_age_greater_84', 'num_white', 'num_african_am', 'num_api', 'num_hispanic',
           'num_native_am', 'num_other_race', 'num_female', 'num_male']
    demo_df = pd.DataFrame(rows_heatmap, columns=col)
    demo_corr = demo_df.corr()

    sns.set(style='white')
    mask = np.zeros_like(demo_corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(16, 12))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    heatmap_plot = sns.heatmap(demo_corr, mask=mask, cmap=cmap, ax=ax)
    heatmap_path = os.path.join(get_abs_path(), 'static', 'tmp', 'heatmap_demo.png')
    heatmap_plot.figure.savefig(heatmap_path, transparent=True)
    return render_template("cost_demo.html", costs_age=costs_age, age_ratio=age_ratio,
                           costs_race=costs_race, race_ratio=race_ratio,
                           costs_sex=costs_sex, sex_ratio=sex_ratio,
                           age_file=url_for('static', filename='tmp/cost_age.csv'),
                           race_file=url_for('static', filename='tmp/cost_race.csv'),
                           sex_file=url_for('static', filename='tmp/cost_sex.csv'),
                           heatmap_fig=url_for('static',
                                       filename='tmp/heatmap_demo.png'))


@app.route('/data')
def data():

    # db_file = get_db()
    # conn = sqlite3.connect(db_file)  # create/open database
    # conn.row_factory=sqlite3.Row
    #
    # # with conn:
    # #     c = conn.cursor()
    # #
    # #     c.execute('''SELECT * FROM report''')
    # #     rows = c.fetchall()

    # columns = ["npi", "provider_last_name", "provider_first_name", "provider_middle_initial", "provider_credentials",
    #            "provider_gender", "provider_entity_type", "provider_street_address_1", "provider_street_address_2",
    #            "provider_city",
    #            "provider_zip_code", "provider_state_code", "provider_country_code", "provider_type",
    #            "medicare_participation_indicator",
    #            "number_of_HCPCS", "number_of_services", "number_of_medicare_beneficiaries",
    #            "total_submitted_charge_amount",
    #            "total_medicare_allowed_amount", "total_medicare_payment_amount",
    #            "total_medicare_standardized_payment_amount",
    #            "drug_suppress_indicator", "number_of_HCPCS_associated_with_drug_services", "number_of_drug_services",
    #            "number_of_medicare_beneficiaries_with_drug_services", "total_drug_submitted_charge_amount",
    #            "total_drug_medicare_allowed_amount",
    #            "total_drug_medicare_payment_amount", "total_drug_medicare_standardized_payment_amount",
    #            "medical_suppress_indicator",
    #            "number_of_HCPCS_associated_medical_services", "number_of_medical_services",
    #            "number_of_medicare_beneficiaries_with_medical_services",
    #            "total_medical_submitted_charge_amount", "total_medical_medicare_allowed_amount",
    #            "total_medical_medicare_payment_amount",
    #            "total_medical_medicare_standardized_payment_amount", "average_age_of_beneficiaries",
    #            "number_of_beneficiaries_age_less_65",
    #            "number_of_beneficiaries_age_65_to_74", "number_of_beneficiaries_age_75_to_84",
    #            "number_of_beneficiaries_age_greater_84",
    #            "number_of_female_beneficiaries", "number_of_male_beneficiaries",
    #            "number_of_non_hispanic_white_beneficiaries",
    #            "number_of_african_american_beneficiaries", "number_of_asian_pacific_islander_beneficiaries",
    #            "number_of_hispanic_beneficiaries",
    #            "number_of_american_indian_alaskan_native_beneficiaries",
    #            "number_of_beneficiaries_with_race_not_elsewhere_classified",
    #            "number_of_beneficiaries_with_medicare_only_entitlement",
    #            "number_of_beneficiaries_with_medicare_and_medicaid_entitlement",
    #            "percent_of_beneficiaries_identified_with_atrial_fibrillation",
    #            "percent_of_beneficiaries_identified_with_alzheimers_disease_or_dementia",
    #            "percent_of_beneficiaries_identified_with_asthma", "percent_of_beneficiaries_identified_with_cancer",
    #            "percent_of_beneficiaries_identified_with_heart_failure",
    #            "percent_of_beneficiaries_identified_with_chronic_kidney_disease",
    #            "percent_of_beneficiaries_identified_with_chronic_obstructive_pulmonary_disease",
    #            "percent_of_beneficiaries_identified_with_depression",
    #            "percent_of_beneficiaries_identified_with_diabetes",
    #            "percent_of_beneficiaries_identified_with_hyperlipidemia",
    #            "percent_of_beneficiaries_identified_with_hypertension",
    #            "percent_of_beneficiaries_identified_with_ischemic_heart_disease",
    #            "percent_of_beneficiaries_identified_with_osteoporosis",
    #            "percent_of_beneficiaries_identified_with_rheumatoid_arthritis_osteoarthritis",
    #            "percent_of_beneficiaries_identified_with_schizophrenia_other_psychotic_disorders",
    #            "percent_of_beneficiaries_identified_with_stroke",
    #            "average_HCC_risk_score_of_beneficiaries"]
    #
    # df=pd.read_sql('SELECT * FROM report;', conn, columns=columns)
    # data = df.head().as_matrix().astype(str)

    return render_template("data.html", cms_img=url_for('static', filename='img/cms_logo.jpg'),
                           bchc_img=url_for('static', filename='img/bch_logo.png'))
    # df = df.head() #head - dataframe
    # data = json.loads(df.to_json()) #exports data frame as json string --> load/parsed json into python object (dict or lsit)
    # return jsonify(data)


@app.route('/data/report')
def report():
    return render_template("report_data.html")


@app.route('/data/puf')
def puf():
    return render_template("puf_data.html")


@app.route('/data/cancer')
def cancer():
    return render_template("cancer_data.html")

# @app.route('/cluster')
# def cluster():
#     # data = db.session.query(Report.npi).first()
#     # data = Report.query.all()
#     s = select([Cancer])
#     conn = db.engine.connect()
#     result=conn.execute(s)
#
#     data= pd.DataFrame(list(result)).select_dtypes(exclude=['object']).dropna().as_matrix()
#
#     scaler = StandardScaler().fit(data)
#     scaled = scaler.transform(data)
#     # PCA
#     pcomp = decomposition.PCA(n_components=2)
#     pcomp.fit(scaled)
#     print pcomp.n_components_
#     components = pcomp.transform(scaled)
#     var = pcomp.explained_variance_ratio_.sum()  # View explained var w/ debug
#     # Kmeans
#     model = KMeans(n_clusters=2)
#     model.fit(components)
#     labels = model.labels_
#     print var
#     # Plot
#     fig = plt.figure()
#     plt.scatter(components[:, 0], components[:, 1], c=model.labels_)
#     centers = plt.plot(
#         [model.cluster_centers_[0, 0], model.cluster_centers_[1, 0]],
#         [model.cluster_centers_[1, 0], model.cluster_centers_[1, 1]],
#         'kx', c='Green'
#     )
#     # Increse size of center points
#     plt.setp(centers, ms=11.0)
#     plt.setp(centers, mew=1.8)
#     # Plot axes adjustments
#     axes = plt.gca()
#     axes.set_xlim([-7.5, 3])
#     axes.set_ylim([-2, 5])
#     plt.xlabel('PC1')
#     plt.ylabel('PC2')
#     plt.title('Clustering of PCs ({:.2f}% Var. Explained)'.format(
#         var * 100
#     ))
#     #Save fig
#     fig_path = os.path.join(get_abs_path(), 'static', 'tmp', 'cluster.png')
#     fig.savefig(fig_path)
#     return render_template('cluster.html',
#                            fig=url_for('static',
#                                        filename='tmp/cluster.png'))
#
#
#     # # Generate CSV
#     # # cluster_data = pd.DataFrame({'pc1': components[:, 0],
#     # #                              'pc2': components[:, 1],
#     # #                              'labels': model.labels_})
#
#     # #Data normalization/scaling
#     # scaler = StandardScaler().fit(data) #scaler object
#     # norm_data = scaler.transform(data) #transformed normalized data
#     #
#     # model = KMeans(n_clusters=2)  # instance of k-means clustering model
#     # model = model.fit(norm_data)  # Fit model to normalized data to provide cluster labeling of data
#     # n_clusters = model.n_clusters  # number of clusters
#     # labels = model.labels_  # cluster labels based on normalized data
#     # Filter original data by cluster labels fitted from normalized data
#     # lst_orig = []  # Set up accumulator for arrays of clusters for original data
#     # for i in range(n_clusters):
#     #     cluster_array = data[labels == i]  # Filter original data for specified cluster label from norm data
#     #     lst_orig += [cluster_array]  # Accumulate filtered array of specified cluster label
#     # # sil_score = silhouette_score(norm_data, labels)  # Silhouette Score
#     #
#     # print lst_orig[0]
#     # fig = plt.figure()
#     # plt.scatter(lst_orig[0][:, 0], lst_orig[0][:, 1], c='b', marker='x')
#     # plt.scatter(lst_orig[1][:, 0], lst_orig[1][:, 1], c='r', marker='^')
#     #
#     # # Plot axes adjustments
#     # plt.xlabel('PC1')
#     # plt.ylabel('PC2')
#     # #Save fig
#     # fig_path = os.path.join(get_abs_path(), 'static', 'tmp', 'cluster.png')
#     # fig.savefig(fig_path)
#     # return render_template('cluster.html',
#     #                        fig=url_for('static',
#     #                                    filename='tmp/cluster.png'))



##main##

