import sqlite3
import os
import pandas as pd
import numpy as np
from pyzipcode import Pyzipcode as pz
from sqlalchemy import Column, ForeignKey, Integer, String, Float, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, scoped_session
from api import db



def get_path():
    f_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'api', 'dataset')
    return f_name


def readCSV():
    columns = ["npi", "provider_last_name", "provider_first_name", "provider_middle_initial", "provider_credentials",
               "provider_gender", "provider_entity_type", "provider_street_address_1", "provider_street_address_2",
               "provider_city", "provider_zip_code", "provider_state_code", "provider_country_code", "provider_type",
               "medicare_participation_indicator", "number_of_HCPCS", "number_of_services", "number_of_medicare_beneficiaries",
               "total_submitted_charge_amount", "total_medicare_allowed_amount", "total_medicare_payment_amount",
               "total_medicare_standardized_payment_amount", "drug_suppress_indicator",
               "number_of_HCPCS_associated_with_drug_services", "number_of_drug_services",
               "number_of_medicare_beneficiaries_with_drug_services", "total_drug_submitted_charge_amount",
               "total_drug_medicare_allowed_amount", "total_drug_medicare_payment_amount", "total_drug_medicare_standardized_payment_amount",
               "medical_suppress_indicator", "number_of_HCPCS_associated_medical_services", "number_of_medical_services",
               "number_of_medicare_beneficiaries_with_medical_services", "total_medical_submitted_charge_amount",
               "total_medical_medicare_allowed_amount", "total_medical_medicare_payment_amount",
               "total_medical_medicare_standardized_payment_amount", "average_age_of_beneficiaries", "number_of_beneficiaries_age_less_65",
               "number_of_beneficiaries_age_65_to_74", "number_of_beneficiaries_age_75_to_84", "number_of_beneficiaries_age_greater_84",
               "number_of_female_beneficiaries", "number_of_male_beneficiaries", "number_of_non_hispanic_white_beneficiaries",
               "number_of_african_american_beneficiaries", "number_of_asian_pacific_islander_beneficiaries",
               "number_of_hispanic_beneficiaries", "number_of_american_indian_alaskan_native_beneficiaries",
               "number_of_beneficiaries_with_race_not_elsewhere_classified", "number_of_beneficiaries_with_medicare_only_entitlement",
               "number_of_beneficiaries_with_medicare_and_medicaid_entitlement", "percent_of_beneficiaries_identified_with_atrial_fibrillation",
               "percent_of_beneficiaries_identified_with_alzheimers_disease_or_dementia",
               "percent_of_beneficiaries_identified_with_asthma", "percent_of_beneficiaries_identified_with_cancer",
               "percent_of_beneficiaries_identified_with_heart_failure", "percent_of_beneficiaries_identified_with_chronic_kidney_disease",
               "percent_of_beneficiaries_identified_with_chronic_obstructive_pulmonary_disease", "percent_of_beneficiaries_identified_with_depression",
               "percent_of_beneficiaries_identified_with_diabetes", "percent_of_beneficiaries_identified_with_hyperlipidemia",
               "percent_of_beneficiaries_identified_with_hypertension", "percent_of_beneficiaries_identified_with_ischemic_heart_disease",
               "percent_of_beneficiaries_identified_with_osteoporosis", "percent_of_beneficiaries_identified_with_rheumatoid_arthritis_osteoarthritis",
               "percent_of_beneficiaries_identified_with_schizophrenia_other_psychotic_disorders",
               "percent_of_beneficiaries_identified_with_stroke", "average_HCC_risk_score_of_beneficiaries"]
    # types = {'npi': np.int64, 'provider_last_name': 'S20', 'provider_first_name': 'S20',
    #          'provider_middle_initial': 'S20', 'provider_credentials': 'S20',
    #          'provider_gender': 'S20', 'provider_entity_type': 'S20', 'provider_street_address_1': 'S20',
    #          'provider_street_address_2': 'S20',
    #          'provider_city': 'S20', 'provider_zip_code': 'S20', 'provider_state_code': 'S20',
    #          'provider_country_code': 'S20', 'provider_type': 'S20',
    #          'medicare_participation_indicator': 'S20', 'number_of_HCPCS': np.int64, 'number_of_services': np.float64,
    #          'number_of_medicare_beneficiaries': np.int64,
    #          'total_submitted_charge_amount': np.float64, 'total_medicare_allowed_amount': np.float64,
    #          'total_medicare_payment_amount': np.float64,
    #          'total_medicare_standardized_payment_amount': np.float64, 'drug_suppress_indicator': 'S20',
    #          'number_of_HCPCS_associated_with_drug_services': np.float64,
    #          'number_of_drug_services': np.float64, 'number_of_medicare_beneficiaries_with_drug_services': np.float64,
    #          'total_drug_submitted_charge_amount': np.float64,
    #          'total_drug_medicare_allowed_amount': np.float64, 'total_drug_medicare_payment_amount': np.float64,
    #          'total_drug_medicare_standardized_payment_amount': np.float64,
    #          'medical_suppress_indicator': 'S20', 'number_of_HCPCS_associated_medical_services': np.float64,
    #          'number_of_medical_services': np.float64,
    #          'number_of_medicare_beneficiaries_with_medical_services': np.float64,
    #          'total_medical_submitted_charge_amount': np.float64,
    #          'total_medical_medicare_allowed_amount': np.float64, 'total_medical_medicare_payment_amount': np.float64,
    #          'total_medical_medicare_standardized_payment_amount': np.float64,
    #          'average_age_of_beneficiaries': np.int64, 'number_of_beneficiaries_age_less_65': np.float64,
    #          'number_of_beneficiaries_age_65_to_74': np.float64,
    #          'number_of_beneficiaries_age_75_to_84': np.float64, 'number_of_beneficiaries_age_greater_84': np.float64,
    #          'number_of_female_beneficiaries': np.float64,
    #          'number_of_male_beneficiaries': np.float64, 'number_of_non_hispanic_white_beneficiaries': np.float64,
    #          'number_of_african_american_beneficiaries': np.float64,
    #          'number_of_asian_pacific_islander_beneficiaries': np.float64,
    #          'number_of_hispanic_beneficiaries': np.float64,
    #          'number_of_american_indian_alaskan_native_beneficiaries': np.float64,
    #          'number_of_beneficiaries_with_race_not_elsewhere_classified': np.float64,
    #          'number_of_beneficiaries_with_medicare_only_entitlement': np.float64,
    #          'number_of_beneficiaries_with_medicare_and_medicaid_entitlement': np.float64,
    #          'percent_of_beneficiaries_identified_with_atrial_fibrillation': np.float64,
    #          'percent_of_beneficiaries_identified_with_alzheimers_disease_or_dementia': np.float64,
    #          'percent_of_beneficiaries_identified_with_asthma': np.float64,
    #          'percent_of_beneficiaries_identified_with_cancer': np.float64,
    #          'percent_of_beneficiaries_identified_with_heart_failure': np.float64,
    #          'percent_of_beneficiaries_identified_with_chronic_kidney_disease': np.float64,
    #          'percent_of_beneficiaries_identified_with_chronic_obstructive_pulmonary_disease': np.float64,
    #          'percent_of_beneficiaries_identified_with_depression': np.float64,
    #          'percent_of_beneficiaries_identified_with_diabetes': np.float64,
    #          'p-ercent_of_beneficiaries_identified_with_hyperlipidemia': np.float64,
    #          'percent_of_beneficiaries_identified_with_hypertension': np.float64,
    #          'percent_of_beneficiaries_identified_with_ischemic_heart_disease': np.float64,
    #          'percent_of_beneficiaries_identified_with_osteoporosis': np.float64,
    #          'percent_of_beneficiaries_identified_with_rheumatoid_arthritis_osteoarthritis': np.float64,
    #          'percent_of_beneficiaries_identified_with_schizophrenia_other_psychotic_disorders': np.float64,
    #          'percent_of_beneficiaries_identified_with_stroke': np.float64,
    #          'average_HCC_risk_score_of_beneficiaries': np.float64}
    types = [('npi', np.int64), ('provider_last_name', 'S20'), ('provider_first_name', 'S20'),
             ('provider_middle_initial', 'S20'), ('provider_credentials', 'S20'),
             ('provider_gender', 'S20'), ('provider_entity_type', 'S20'), ('provider_street_address_1', 'S20'),
             ('provider_street_address_2', 'S20'),
             ('provider_city', 'S20'), ('provider_zip_code', 'S20'), ('provider_state_code', 'S20'),
             ('provider_country_code', 'S20'), ('provider_type', 'S20'),
             ('medicare_participation_indicator', 'S20'), ('number_of_HCPCS', np.int64), ('number_of_services', np.float64),
             ('number_of_medicare_beneficiaries', np.int64),
             ('total_submitted_charge_amount', np.float64), ('total_medicare_allowed_amount', np.float64),
             ('total_medicare_payment_amount', np.float64),
             ('total_medicare_standardized_payment_amount', np.float64), ('drug_suppress_indicator', 'S20'),
             ('number_of_HCPCS_associated_with_drug_services', np.float64),
             ('number_of_drug_services', np.float64), ('number_of_medicare_beneficiaries_with_drug_services', np.float64),
             ('total_drug_submitted_charge_amount', np.float64),
             ('total_drug_medicare_allowed_amount', np.float64), ('total_drug_medicare_payment_amount', np.float64),
             ('total_drug_medicare_standardized_payment_amount', np.float64),
             ('medical_suppress_indicator', 'S20'), ('number_of_HCPCS_associated_medical_services', np.float64),
             ('number_of_medical_services', np.float64),
             ('number_of_medicare_beneficiaries_with_medical_services', np.float64),
             ('total_medical_submitted_charge_amount', np.float64),
             ('total_medical_medicare_allowed_amount', np.float64), ('total_medical_medicare_payment_amount', np.float64),
             ('total_medical_medicare_standardized_payment_amount', np.float64),
             ('average_age_of_beneficiaries', np.int64), ('number_of_beneficiaries_age_less_65', np.float64),
             ('number_of_beneficiaries_age_65_to_74', np.float64),
             ('number_of_beneficiaries_age_75_to_84', np.float64), ('number_of_beneficiaries_age_greater_84', np.float64),
             ('number_of_female_beneficiaries', np.float64),
             ('number_of_male_beneficiaries', np.float64), ('number_of_non_hispanic_white_beneficiaries', np.float64),
             ('number_of_african_american_beneficiaries', np.float64),
             ('number_of_asian_pacific_islander_beneficiaries', np.float64),
             ('number_of_hispanic_beneficiaries', np.float64),
             ('number_of_american_indian_alaskan_native_beneficiaries', np.float64),
             ('number_of_beneficiaries_with_race_not_elsewhere_classified', np.float64),
             ('number_of_beneficiaries_with_medicare_only_entitlement', np.float64),
             ('number_of_beneficiaries_with_medicare_and_medicaid_entitlement', np.float64),
             ('percent_of_beneficiaries_identified_with_atrial_fibrillation', np.float64),
             ('percent_of_beneficiaries_identified_with_alzheimers_disease_or_dementia', np.float64),
             ('percent_of_beneficiaries_identified_with_asthma', np.float64),
             ('percent_of_beneficiaries_identified_with_cancer', np.float64),
             ('percent_of_beneficiaries_identified_with_heart_failure', np.float64),
             ('percent_of_beneficiaries_identified_with_chronic_kidney_disease', np.float64),
             ('percent_of_beneficiaries_identified_with_chronic_obstructive_pulmonary_disease', np.float64),
             ('percent_of_beneficiaries_identified_with_depression', np.float64),
             ('percent_of_beneficiaries_identified_with_diabetes', np.float64),
             ('percent_of_beneficiaries_identified_with_hyperlipidemia', np.float64),
             ('percent_of_beneficiaries_identified_with_hypertension', np.float64),
             ('percent_of_beneficiaries_identified_with_ischemic_heart_disease', np.float64),
             ('percent_of_beneficiaries_identified_with_osteoporosis', np.float64),
             ('percent_of_beneficiaries_identified_with_rheumatoid_arthritis_osteoarthritis', np.float64),
             ('percent_of_beneficiaries_identified_with_schizophrenia_other_psychotic_disorders', np.float64),
             ('percent_of_beneficiaries_identified_with_stroke', np.float64),
             ('average_HCC_risk_score_of_beneficiaries', np.float64)]
    df = pd.read_csv(os.path.join(get_path(),
                                    'Medicare_Physician_and_Other_Supplier_National_Provider_Identifier__NPI__Aggregate_Report__Calendar_Year_2014.csv'),
                       sep=',', names=columns, header=0, dtype=types)

    #filter for only US states -- Convert to Numpy array
    state = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA',
             'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK',
             'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC']

    terr = ['PR', 'GU', 'VI', 'AS', 'District of Columbia', 'MP', 'AA', 'AE', 'AP'] #USA territories
    usa = state+terr
    data=df.as_matrix()
    US_data = np.array([row for row in data if row[12]=='US'])
    for row in US_data:
        if row[11] not in usa and len(str(row[10])) >=5:
            location=pz.get(int(str(row[10])[:5]),'US')
            if location != False:
                row[10]=location['postal_code'] #correct ZIP code
                row[11]=location['state_short'] #correct state code
    state_data= np.array([row for row in US_data if row[11] in state])

    #Convert to recarray -- transfer hetergeneous column dtypes to DataFrame
    state_recarray = np.core.records.fromarrays(np.transpose(state_data), dtype=types, names=columns)
    #Convert to Pandas DataFrame
    # state_df = pd.DataFrame(data=state_data, columns=columns)
    # state_df = state_df.convert_objects(convert_numeric=True)

    state_df = pd.DataFrame.from_records(state_recarray, columns=columns)
    state_df = state_df.replace(to_replace='', value=np.nan)
    return state_df


def create_table():

    db_file = os.path.join(get_path(), 'cms.db')
    schema_file = os.path.join(get_path(), 'cms_schema.sql')
    update_file = os.path.join(get_path(), 'update.sql')

    db_is_new = not os.path.exists(db_file)
    #Connecting/Creating database file
    conn = sqlite3.connect(db_file)
    c = conn.cursor()

    if db_is_new:
        print "Database created, creating table(s) schema"
        print "Reading SQL script..."
        f_schema = open(schema_file, 'r')
        schema = f_schema.read()
        f_schema.close()
        c.executescript(schema) #create schema
        #import data into SQL db
        conn.text_factory = lambda x: unicode(x, 'utf-8', 'ignore')
        data = readCSV()
        cms = data.to_sql('report', con=conn, flavor='sqlite', if_exists='append', index=False)
        #Update empty strings into SQL NULL values
        f_update = open(update_file, 'r')
        update = f_update.read()
        f_update.close()
        c.executescript(update)

    else:
        print "Database exists; opened successfully"

    conn.commit()
    return "Database Initialization complete"


def query(path):

    db_file = os.path.join(path, 'cms.db')
    queries_file = os.path.join(path, "cms_queries.sql")
    print queries_file

    conn = sqlite3.connect(db_file) #create/open database
    print "Connected to database successfully"
    with conn:

        c = conn.cursor()

        print "Reading SQL script..."
        f_queries = open(queries_file, 'r')
        query = f_queries.read()
        f_queries.close()

        query_commands=query.split(';')
        # print query_commands

        print "Running SQL script..."
        query_lst=[]
        for command in query_commands:
            # try:
            c.execute(command)
            row = c.fetchall()
            query_lst+=[row]
            # except OperationalError, msg:
            #     print "Command skipped: ", msg


        # c.execute("SELECT provider_state_code, percent_of_beneficiaries_identified_with_cancer FROM report GROUP BY provider_state_code;")
        # print c.fetchone()
        # rows = c.fetchall()
        # for row in rows:
        #     print row
    # return rows
    return query_lst

Base = declarative_base()
session=scoped_session(sessionmaker())


class Report(Base):
    __tablename__ = "report"

    npi = Column(Integer, primary_key=True)
    provider_last_name = Column(String(50), nullable=True)
    provider_first_name = Column(String(50), nullable=True)
    provider_middle_initial = Column(String(50), nullable=True)
    provider_credentials = Column(String(50), nullable=True)
    provider_gender = Column(String(50), nullable=True)
    provider_entity_type = Column(String(50), nullable=True)
    provider_street_address_1 = Column(String(50), nullable=True)
    provider_street_address_2 = Column(String(50), nullable=True)
    provider_city = Column(String(50), nullable=True)
    provider_zip_code = Column(Integer, nullable=True)
    provider_state_code = Column(String(50), nullable=True)
    provider_country_code = Column(String(50), nullable=True)
    provider_type = Column(String(50), nullable=True)
    medicare_participation_indicator = Column(String(50), nullable=True)
    number_of_HCPCS = Column(Integer, nullable=True)
    number_of_services = Column(Integer, nullable=True)
    number_of_medicare_beneficiaries = Column(Integer, nullable=True)
    total_submitted_charge_amount = Column(Float, nullable=True)
    total_medicare_allowed_amount = Column(Float, nullable=True)
    total_medicare_payment_amount = Column(Float, nullable=True)
    total_medicare_standardized_payment_amount = Column(Float, nullable=True)
    drug_suppress_indicator = Column(String(50), nullable=True)
    number_of_HCPCS_associated_with_drug_services = Column(Integer, nullable=True)
    number_of_drug_services = Column(Integer, nullable=True)
    number_of_medicare_beneficiaries_with_drug_services = Column(Integer, nullable=True)
    total_drug_submitted_charge_amount = Column(Float, nullable=True)
    total_drug_medicare_allowed_amount = Column(Float, nullable=True)
    total_drug_medicare_payment_amount = Column(Float, nullable=True)
    total_drug_medicare_standardized_payment_amount = Column(Float, nullable=True)
    medical_suppress_indicator = Column(String(50), nullable=True)
    number_of_HCPCS_associated_medical_services = Column(Integer, nullable=True)
    number_of_medical_services = Column(Integer, nullable=True)
    number_of_medicare_beneficiaries_with_medical_services = Column(Integer, nullable=True)
    total_medical_submitted_charge_amount = Column(Float, nullable=True)
    total_medical_medicare_allowed_amount = Column(Float, nullable=True)
    total_medical_medicare_payment_amount = Column(Float, nullable=True)
    total_medical_medicare_standardized_payment_amount = Column(Float, nullable=True)
    average_age_of_beneficiaries = Column(Integer, nullable=True)
    number_of_beneficiaries_age_less_65 = Column(Integer, nullable=True)
    number_of_beneficiaries_age_65_to_74 = Column(Integer, nullable=True)
    number_of_beneficiaries_age_75_to_84 = Column(Integer, nullable=True)
    number_of_beneficiaries_age_greater_84 = Column(Integer, nullable=True)
    number_of_female_beneficiaries = Column(Integer, nullable=True)
    number_of_male_beneficiaries = Column(Integer, nullable=True)
    number_of_non_hispanic_white_beneficiaries = Column(Integer, nullable=True)
    number_of_african_american_beneficiaries = Column(Integer, nullable=True)
    number_of_asian_pacific_islander_beneficiaries = Column(Integer, nullable=True)
    number_of_hispanic_beneficiaries = Column(Integer, nullable=True)
    number_of_american_indian_alaskan_native_beneficiaries = Column(Integer, nullable=True)
    number_of_beneficiaries_with_race_not_elsewhere_classified = Column(Integer, nullable=True)
    number_of_beneficiaries_with_medicare_only_entitlement = Column(Integer, nullable=True)
    number_of_beneficiaries_with_medicare_and_medicaid_entitlement = Column(Integer, nullable=True)
    percent_of_beneficiaries_identified_with_atrial_fibrillation = Column(Integer, nullable=True)
    percent_of_beneficiaries_identified_with_alzheimers_disease_or_dementia = Column(Integer, nullable=True)
    percent_of_beneficiaries_identified_with_asthma = Column(Integer, nullable=True)
    percent_of_beneficiaries_identified_with_cancer = Column(Integer, nullable=True)
    percent_of_beneficiaries_identified_with_heart_failure = Column(Integer, nullable=True)
    percent_of_beneficiaries_identified_with_chronic_kidney_disease = Column(Integer, nullable=True)
    percent_of_beneficiaries_identified_with_chronic_obstructive_pulmonary_disease = Column(Integer, nullable=True)
    percent_of_beneficiaries_identified_with_depression = Column(Integer, nullable=True)
    percent_of_beneficiaries_identified_with_diabetes = Column(Integer, nullable=True)
    percent_of_beneficiaries_identified_with_hyperlipidemia = Column(Integer, nullable=True)
    percent_of_beneficiaries_identified_with_hypertension = Column(Integer, nullable=True)
    percent_of_beneficiaries_identified_with_ischemic_heart_disease = Column(Integer, nullable=True)
    percent_of_beneficiaries_identified_with_osteoporosis = Column(Integer, nullable=True)
    percent_of_beneficiaries_identified_with_rheumatoid_arthritis_osteoarthritis = Column(Integer, nullable=True)
    percent_of_beneficiaries_identified_with_schizophrenia_other_psychotic_disorders = Column(Integer, nullable=True)
    percent_of_beneficiaries_identified_with_stroke = Column(Integer, nullable=True)
    average_HCC_risk_score_of_beneficiaries = Column(Float, nullable=True)



def init_db():

    # #Create engine to store data in local directory's db file
    db_path=os.path.join(get_path(), 'cms3.db')
    engine = db.engine # sqlalchemy lib -- engine=create_engine('sqlite:///%s' % (db_path))
    db_is_new = not os.path.exists(db_path)
    if db_is_new:
        print "Database created, creating table(s) schema"
        #Remove spontaneous quoting of column name
        db.engine.dialect.identifier_preparer.initial_quote = ''
        db.engine.dialect.identifier_preparer.final_quote = ''

        #Create schema -- all tables in the engine -- equivalent to SQL "Create Table"
        db.create_all() # sqlalchemy lib -- Base.metadata.create_all(bind=engine)

        #Insert Data
        #Individual Insert
        # db.session.add(Report(npi=110, provider_state_code='AZ'))
        # db.session.commit()

        # Bulk insert of DataFrame
        df = readCSV()
        df_lst = df.to_dict(orient='records')  # orient by records to align format
        db.session.execute(Report.__table__.insert(), df_lst)
        db.session.commit() #Commit

        # metadata = sqlalchemy.schema.MetaData(bind=engine, reflect=True)
        # table = sqlalchemy.Table('report', metadata, autoload=True)
        #
        # session.execute(table.insert(), df_lst)
        #
        # session.commit()  # commit changes
        # session.close()  # close session
    else:
        print "Database exists; opened successfully"
    return


# def insert_db():
#     engine = init_db()



#main
# data = readCSV()
# print data.to_dict(orient='records')[0]
# print data["provider_middle_initial"][1], data["provider_middle_initial"][1]
# print get_path()
# print create_table()
# print query(get_path())

init_db()

# print data.isnull().sum()
# print data.dropna()
# print data.shape
# print data.dropna().dtypes
