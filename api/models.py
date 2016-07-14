from api import app
from api import db


class Report(db.Model):
    __tablename__ = "report"

    npi = db.Column(db.Integer, primary_key=True)
    provider_last_name = db.Column(db.String(50), nullable=True)
    provider_first_name = db.Column(db.String(50), nullable=True)
    provider_middle_initial = db.Column(db.String(50), nullable=True)
    provider_credentials = db.Column(db.String(50), nullable=True)
    provider_gender = db.Column(db.String(50), nullable=True)
    provider_entity_type = db.Column(db.String(50), nullable=True)
    provider_street_address_1 = db.Column(db.String(50), nullable=True)
    provider_street_address_2 = db.Column(db.String(50), nullable=True)
    provider_city = db.Column(db.String(50), nullable=True)
    provider_zip_code = db.Column(db.Integer, nullable=True)
    provider_state_code = db.Column(db.String(50), nullable=True)
    provider_country_code = db.Column(db.String(50), nullable=True)
    provider_type = db.Column(db.String(50), nullable=True)
    medicare_participation_indicator = db.Column(db.String(50), nullable=True)
    number_of_HCPCS = db.Column(db.Integer, nullable=True)
    number_of_services = db.Column(db.Integer, nullable=True)
    number_of_medicare_beneficiaries = db.Column(db.Integer, nullable=True)
    total_submitted_charge_amount = db.Column(db.Float, nullable=True)
    total_medicare_allowed_amount = db.Column(db.Float, nullable=True)
    total_medicare_payment_amount = db.Column(db.Float, nullable=True)
    total_medicare_standardized_payment_amount = db.Column(db.Float, nullable=True)
    drug_suppress_indicator = db.Column(db.String(50), nullable=True)
    number_of_HCPCS_associated_with_drug_services = db.Column(db.Integer, nullable=True)
    number_of_drug_services = db.Column(db.Integer, nullable=True)
    number_of_medicare_beneficiaries_with_drug_services = db.Column(db.Integer, nullable=True)
    total_drug_submitted_charge_amount = db.Column(db.Float, nullable=True)
    total_drug_medicare_allowed_amount = db.Column(db.Float, nullable=True)
    total_drug_medicare_payment_amount = db.Column(db.Float, nullable=True)
    total_drug_medicare_standardized_payment_amount = db.Column(db.Float, nullable=True)
    medical_suppress_indicator = db.Column(db.String(50), nullable=True)
    number_of_HCPCS_associated_medical_services = db.Column(db.Integer, nullable=True)
    number_of_medical_services = db.Column(db.Integer, nullable=True)
    number_of_medicare_beneficiaries_with_medical_services = db.Column(db.Integer, nullable=True)
    total_medical_submitted_charge_amount = db.Column(db.Float, nullable=True)
    total_medical_medicare_allowed_amount = db.Column(db.Float, nullable=True)
    total_medical_medicare_payment_amount = db.Column(db.Float, nullable=True)
    total_medical_medicare_standardized_payment_amount = db.Column(db.Float, nullable=True)
    average_age_of_beneficiaries = db.Column(db.Integer, nullable=True)
    number_of_beneficiaries_age_less_65 = db.Column(db.Integer, nullable=True)
    number_of_beneficiaries_age_65_to_74 = db.Column(db.Integer, nullable=True)
    number_of_beneficiaries_age_75_to_84 = db.Column(db.Integer, nullable=True)
    number_of_beneficiaries_age_greater_84 = db.Column(db.Integer, nullable=True)
    number_of_female_beneficiaries = db.Column(db.Integer, nullable=True)
    number_of_male_beneficiaries = db.Column(db.Integer, nullable=True)
    number_of_non_hispanic_white_beneficiaries = db.Column(db.Integer, nullable=True)
    number_of_african_american_beneficiaries = db.Column(db.Integer, nullable=True)
    number_of_asian_pacific_islander_beneficiaries = db.Column(db.Integer, nullable=True)
    number_of_hispanic_beneficiaries = db.Column(db.Integer, nullable=True)
    number_of_american_indian_alaskan_native_beneficiaries = db.Column(db.Integer, nullable=True)
    number_of_beneficiaries_with_race_not_elsewhere_classified = db.Column(db.Integer, nullable=True)
    number_of_beneficiaries_with_medicare_only_entitlement = db.Column(db.Integer, nullable=True)
    number_of_beneficiaries_with_medicare_and_medicaid_entitlement = db.Column(db.Integer, nullable=True)
    percent_of_beneficiaries_identified_with_atrial_fibrillation = db.Column(db.Integer, nullable=True)
    percent_of_beneficiaries_identified_with_alzheimers_disease_or_dementia = db.Column(db.Integer, nullable=True)
    percent_of_beneficiaries_identified_with_asthma = db.Column(db.Integer, nullable=True)
    percent_of_beneficiaries_identified_with_cancer = db.Column(db.Integer, nullable=True)
    percent_of_beneficiaries_identified_with_heart_failure = db.Column(db.Integer, nullable=True)
    percent_of_beneficiaries_identified_with_chronic_kidney_disease = db.Column(db.Integer, nullable=True)
    percent_of_beneficiaries_identified_with_chronic_obstructive_pulmonary_disease = db.Column(db.Integer, nullable=True)
    percent_of_beneficiaries_identified_with_depression = db.Column(db.Integer, nullable=True)
    percent_of_beneficiaries_identified_with_diabetes = db.Column(db.Integer, nullable=True)
    percent_of_beneficiaries_identified_with_hyperlipidemia = db.Column(db.Integer, nullable=True)
    percent_of_beneficiaries_identified_with_hypertension = db.Column(db.Integer, nullable=True)
    percent_of_beneficiaries_identified_with_ischemic_heart_disease = db.Column(db.Integer, nullable=True)
    percent_of_beneficiaries_identified_with_osteoporosis = db.Column(db.Integer, nullable=True)
    percent_of_beneficiaries_identified_with_rheumatoid_arthritis_osteoarthritis = db.Column(db.Integer, nullable=True)
    percent_of_beneficiaries_identified_with_schizophrenia_other_psychotic_disorders = db.Column(db.Integer, nullable=True)
    percent_of_beneficiaries_identified_with_stroke = db.Column(db.Integer, nullable=True)
    average_HCC_risk_score_of_beneficiaries = db.Column(db.Float, nullable=True)
    # puf = db.relationship('puf', backref='report')


class Puf(db.Model):
    __tablename__ = "puf"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    npi = db.Column(db.Integer, primary_key=True)
    provider_last_name = db.Column(db.String(50), nullable=True)
    provider_first_name = db.Column(db.String(50), nullable=True)
    provider_middle_initial = db.Column(db.String(50), nullable=True)
    provider_credentials = db.Column(db.String(50), nullable=True)
    provider_gender = db.Column(db.String(50), nullable=True)
    provider_entity_type = db.Column(db.String(50), nullable=True)
    provider_street_address_1 = db.Column(db.String(50), nullable=True)
    provider_street_address_2 = db.Column(db.String(50), nullable=True)
    provider_city = db.Column(db.String(50), nullable=True)
    provider_zip_code = db.Column(db.Integer, nullable=True)
    provider_state_code = db.Column(db.String(50), nullable=True)
    provider_country_code = db.Column(db.String(50), nullable=True)
    provider_type = db.Column(db.String(50), nullable=True)
    medicare_participation_indicator = db.Column(db.String(50), nullable=True)
    place_of_service = db.Column(db.String(50), nullable=True)
    HCPCS_code = db.Column(db.String(50), nullable=True)
    HCPCS_description = db.Column(db.String(50), nullable=True)
    identifies_HCPCS_as_drug_included_in_the_ASP_drug_list = db.Column(db.String(50), nullable=True)
    number_of_services = db.Column(db.Integer, nullable=True)
    number_of_medicare_beneficiaries = db.Column(db.Integer, nullable=True)
    number_of_distinct_medicare_beneficiary_per_day_services = db.Column(db.Integer, nullable=True)
    average_medicare_allowed_amount = db.Column(db.Float, nullable=True)
    average_submitted_charge_amount = db.Column(db.Float, nullable=True)
    average_medicare_payment_amount = db.Column(db.Float, nullable=True)
    average_medicare_standardized_amount = db.Column(db.Float, nullable=True)
    report_npi = db.Column(db.Integer, db.ForeignKey('report.npi'))


class Cancer(db.Model):
    __tablename__ = "cancer"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    indicator = db.Column(db.String(10), nullable=True)
    year = db.Column(db.Integer, nullable=True)
    gender = db.Column(db.String(10), nullable=True)
    race = db.Column(db.String(20), nullable=True)
    value = db.Column(db.Float, nullable=True)
    place = db.Column(db.String(50), nullable=True)


