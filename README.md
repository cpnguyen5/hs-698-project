# HS 698 Project
#### Centers for Medicare & Medicaid Services (CMS): Provider Utilization & Payment Plan
![alt text](http://www.csh.org/wp-content/uploads/2015/06/CMS.jpg "CMS")

## Project Description
##### Web Application Development Exploring Healthcare Utilization under Medicare & Medicaid Programs
### Purpose
Develop a Python Flask web application (lightweight web framework) to investigate the utilization 
of services & procedures and the relative healthcare costs amongst providers participating in 
Medicare & Medicaid programs. The goal of such data exploration is to evaluate the current quality 
of care being provided to beneficiaries under the CMS (Centers for Medicare & Medicaid Services) 
fee-for-serve programs. Essentially, this project is a cross-sectional study to provide a high-level
scope on the variability of healthcare costs in relation to key factors (e.g. demographics, utilized services).
The web application is built upon a SQL database, which may be constantly updated to evaluate the 
quality of care for a given year.

## Deploy Web Application
The Flask web application is deployed through AWS's E2 instance (Ubuntu 14.04 server). The owner must run the 
web application via `python run_api.py` through the terminal within the Ubuntu server. 
*Notes*:
  * URL: http://54.187.154.214:5003
  * The AWS branch of the project encompasses the deployment version of the application.

## Local Deployment (Steps)
1. Clone git repository `git clone <repo>`
2. Create [virtual environment] & activate
3. Install Python application requirements: `pip install -r requirements.txt`
4. cd into root directory of repo within the terminal and run application: `python run_api.py`
5. Launch http://localhost:5003/
[virtual environment]: http://docs.python-guide.org/en/latest/dev/virtualenvs/

## Branches
### Master
The master branch is the final **production** version of the project, encompassing the deployed version of the 
Flask application. The production configuration implements a server-based PostgreSQL database through Amazon
Web Service's (AWS) Relational Database Service (RDS).
*Database*: PostgreSQL **via AWS's RDS**
*Deployment*: AWS E2 Instance (http://54.187.154.214:5003)
  * Public IP: 54.187.154.214
  * Port: 5003
 
### sqlite3
The sqlite3 branch is identical to the master branch and preserves the sqlite3 database configuration.
*Database*: local sqlite3
*Deployment*: local (http://localhost:5003/)
  * Port: 5003

### dev
The dev branch consists of the **developmental** configuration, implementing a local PostgreSQL database.
*Database*: local PostgreSQL
*Deployment*: local (http://localhost:5003/)
  * Port: 5003

### aws
The aws branch consists of the development of the **production** configuration, implementing a server-based 
PostgreSQL database through AWS's Relational Database Service (RDS).
*Database*: PostgreSQL **via AWS's RDS**
*Deployment*: AWS E2 Instance (http://54.187.154.214:5003)
  * Public IP: 54.187.154.214
  * Port: 5003
 
 
## SQL Database
The project uses sqlite3 for development and PostgreSQL for both development and production/deployment purposes.
SQLAlchemy, an object relational mapper, is utilized to efficiently bridge the connection between the underlying
SQL databases and Python code. SQLAlchemy's syntax is implemented for table schema, database insert, and queries.

For more information, please refer to [SQLAlchemy's documentation].
[SQLAlchemy's documentation]: http://docs.sqlalchemy.org/en/latest/


## Machine Learning
Various unsupervised learning approaches were attempted, but after investigation it was determined that clustering
may not be effectively applied onto the CMS dataset.

The following are a few observations from the attempts:
  * Principal Component Analysis (PCA) was implemented to perform dimension reduction or scale the dataset. Unfortunately,
    the top 2-3 principal components did not hold much weight. The cumulative sum of their explained variance ratio was
    < 60%. In summary, scaling the dimensions of the dataset down to a shape that can be visualized and interpreted
    was not effective. The low cumulative explained variance ratio of the top principal components indicates that they
    were unable to capture much of the variance of the data.
  * The labeling by various clustering models (e.g. K-Means, Agglomerative, DBSCAN) was highly discrepant as at least
    1 cluster/label dominated in size (number of labeled points), accounting for > 99% of the data points. Any
    additional clusters appeared to be more similar to outliers, in terms of function. Thus, this dominance of a label
    may indicate low variance amongst the data.
  * The variables of the data appeared to generally have a linear relationship. Thus, clustering may not be the best
    approach for machine learning applications.  